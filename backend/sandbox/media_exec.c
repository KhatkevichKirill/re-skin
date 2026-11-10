/* AICP-287: unprivileged, fail-closed media confinement on Landlock ABI 1.
 * Build executable and shared ASR installer from this same translation unit.
 * No setuid, namespaces, inherited socket descriptors, or executable override.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/landlock.h>
#include <sched.h>
#include <seccomp.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_INPUTS 64
#define MAX_OUTPUTS 4
#define FILE_LIMIT (512ULL * 1024 * 1024)
#define ABI1_RIGHTS ((LANDLOCK_ACCESS_FS_MAKE_SYM << 1) - 1)

static const char *const runtime_paths[] = {
    "/usr/lib", "/usr/lib64", "/lib", "/lib64", "/etc/ld.so.cache",
    "/etc/fonts", "/usr/share/fonts", "/usr/share/fontconfig",
    "/var/cache/fontconfig", "/usr/share/locale", "/etc/localtime", NULL
};

static int approved_tool(const char *tool) {
    return tool && (!strcmp(tool, "/usr/bin/ffmpeg") ||
                    !strcmp(tool, "/usr/bin/ffprobe") ||
                    !strcmp(tool, "/usr/bin/fc-match"));
}

static int add_path(int ruleset, const char *path, uint64_t rights,
                    int optional, int require_regular, int output) {
    struct stat st;
    struct landlock_path_beneath_attr rule = {0};
    int fd = open(path, O_PATH | O_CLOEXEC | (require_regular ? O_NOFOLLOW : 0));
    if (fd < 0) return optional && errno == ENOENT ? 0 : -1;
    if (fstat(fd, &st) || (require_regular && !S_ISREG(st.st_mode)) ||
        (output && (st.st_nlink != 1 || st.st_uid != geteuid()))) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (!S_ISDIR(st.st_mode)) rights &= ~LANDLOCK_ACCESS_FS_READ_DIR;
    rule.parent_fd = fd;
    rule.allowed_access = rights;
    int result = syscall(SYS_landlock_add_rule, ruleset,
                         LANDLOCK_RULE_PATH_BENEATH, &rule, 0);
    int saved = errno;
    close(fd);
    errno = saved;
    return result;
}

static int landlock_policy(const char *tool, const char *const *inputs,
                           size_t input_count, const char *const *outputs,
                           size_t output_count, const char *model_dir) {
    if (input_count > MAX_INPUTS || output_count > MAX_OUTPUTS) return -1;
    if (syscall(SYS_landlock_create_ruleset, NULL, 0,
                LANDLOCK_CREATE_RULESET_VERSION) < 1) return -1;
    struct landlock_ruleset_attr attr = {.handled_access_fs = ABI1_RIGHTS};
    int ruleset = syscall(SYS_landlock_create_ruleset, &attr, sizeof(attr), 0);
    if (ruleset < 0) return -1;
    int result = -1;
    uint64_t read = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR;
    for (size_t i = 0; runtime_paths[i]; ++i)
        if (add_path(ruleset, runtime_paths[i], read, 1, 0, 0)) goto done;
    if (tool) {
        if (!approved_tool(tool) || add_path(ruleset, tool,
            LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_EXECUTE, 0, 1, 0)) goto done;
        /* ELF interpreters are executable checks in execve, never broad /lib EXEC. */
        const char *loaders[] = {"/lib64/ld-linux-x86-64.so.2",
            "/lib/ld-linux-aarch64.so.1", NULL};
        for (size_t i = 0; loaders[i]; ++i)
            if (add_path(ruleset, loaders[i], LANDLOCK_ACCESS_FS_READ_FILE |
                         LANDLOCK_ACCESS_FS_EXECUTE, 1, 0, 0)) goto done;
    }
    if (model_dir) {
        const char *python_paths[] = {"/opt/asr", "/opt/venv", "/usr/local/lib", NULL};
        for (size_t i = 0; python_paths[i]; ++i)
            if (add_path(ruleset, python_paths[i], read, 1, 0, 0)) goto done;
        if (add_path(ruleset, model_dir, read, 0, 0, 0)) goto done;
    }
    for (size_t i = 0; i < input_count; ++i)
        if (!inputs[i] || inputs[i][0] != '/' ||
            add_path(ruleset, inputs[i], LANDLOCK_ACCESS_FS_READ_FILE, 0, 1, 0)) goto done;
    for (size_t i = 0; i < output_count; ++i)
        if (!outputs[i] || outputs[i][0] != '/' ||
            add_path(ruleset, outputs[i], LANDLOCK_ACCESS_FS_READ_FILE |
                     LANDLOCK_ACCESS_FS_WRITE_FILE, 0, 1, 1)) goto done;
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) ||
        syscall(SYS_landlock_restrict_self, ruleset, 0)) goto done;
    result = 0;
done:
    close(ruleset);
    return result;
}

static int allow_name(scmp_filter_ctx context, const char *name) {
    int nr = seccomp_syscall_resolve_name(name);
    return nr == __NR_SCMP_ERROR ? 0 : seccomp_rule_add(context, SCMP_ACT_ALLOW, nr, 0);
}

static int syscall_policy(int may_exec) {
    /* Everything absent here, including future syscalls, fails with EPERM. */
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ERRNO(EPERM));
    if (!ctx) return -1;
    int result = -1;
    const char *allowed[] = {
        "read", "readv", "pread64", "preadv", "preadv2", "write", "writev",
        "pwrite64", "pwritev", "pwritev2", "close", "close_range", "lseek",
        "fstat", "newfstatat", "stat", "lstat", "statx", "statfs", "fstatfs",
        "access", "faccessat", "faccessat2", "readlink", "readlinkat", "getcwd",
        "getdents", "getdents64", "dup", "dup2", "dup3",
        "mmap", "mmap2", "mprotect", "munmap", "mremap", "brk", "madvise",
        "msync", "mincore", "ftruncate", "fsync", "fdatasync", "fallocate",
        "poll", "ppoll", "select", "pselect6", "epoll_create", "epoll_create1",
        "epoll_ctl", "epoll_wait", "epoll_pwait", "epoll_pwait2", "eventfd", "eventfd2",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigaltstack",
        "rt_sigpending", "rt_sigtimedwait", "restart_syscall", "pause",
        "futex", "futex_waitv", "set_robust_list", "rseq",
        "clock_gettime", "clock_getres", "clock_nanosleep", "gettimeofday",
        "nanosleep", "times", "getrusage", "getrlimit",
        "getpid", "getppid", "gettid", "getuid", "geteuid", "getgid", "getegid",
        "getgroups", "getresuid", "getresgid", "sched_yield", "sched_getaffinity",
        "sched_getparam", "sched_getscheduler", "getcpu", "uname", "sysinfo",
        "getrandom", "arch_prctl", "set_tid_address", "exit", "exit_group", NULL
    };
    for (size_t i = 0; allowed[i]; ++i) if (allow_name(ctx, allowed[i])) goto done;
    const int fcntl_commands[] = {F_DUPFD, F_DUPFD_CLOEXEC, F_GETFD, F_SETFD, F_GETFL, F_SETFL};
    for (size_t i = 0; i < sizeof(fcntl_commands)/sizeof(fcntl_commands[0]); ++i)
        if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(fcntl), 1,
            SCMP_A1(SCMP_CMP_EQ, fcntl_commands[i]))) goto done;
    if (may_exec && allow_name(ctx, "execve")) goto done;
    /* ABI 1 does not control truncation: O_RDONLY|O_TRUNC must be refused.
     * O_WRONLY/O_RDWR truncation is mediated by Landlock WRITE_FILE.
     * openat2 is deliberately absent: classic BPF cannot inspect open_how.
     */
    if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(open), 1,
        SCMP_A1(SCMP_CMP_MASKED_EQ, O_TRUNC, 0)) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(open), 1,
        SCMP_A1(SCMP_CMP_MASKED_EQ, O_ACCMODE, O_WRONLY)) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(open), 1,
        SCMP_A1(SCMP_CMP_MASKED_EQ, O_ACCMODE, O_RDWR)) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(openat), 1,
        SCMP_A2(SCMP_CMP_MASKED_EQ, O_TRUNC, 0)) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(openat), 1,
        SCMP_A2(SCMP_CMP_MASKED_EQ, O_ACCMODE, O_WRONLY)) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(openat), 1,
        SCMP_A2(SCMP_CMP_MASKED_EQ, O_ACCMODE, O_RDWR))) goto done;
    /* glibc falls back to clone when clone3 returns ENOSYS. Only pthread
     * clones are admitted: no new processes, namespaces, or signal escapes. */
    unsigned long required = CLONE_VM | CLONE_SIGHAND | CLONE_THREAD;
    unsigned long permitted = required | CLONE_FS | CLONE_FILES | CLONE_SYSVSEM |
        CLONE_SETTLS | CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID | CLONE_CHILD_SETTID;
    if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(clone), 1,
        SCMP_A0(SCMP_CMP_MASKED_EQ, required | ~permitted, required))) goto done;
    int clone3_nr = seccomp_syscall_resolve_name("clone3");
    if (clone3_nr != __NR_SCMP_ERROR && seccomp_rule_add(ctx,
        SCMP_ACT_ERRNO(ENOSYS), clone3_nr, 0)) goto done;
    if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(tgkill), 1,
        SCMP_A0(SCMP_CMP_EQ, (scmp_datum_t)getpid())) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(prlimit64), 1,
        SCMP_A0(SCMP_CMP_EQ, 0)) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(prlimit64), 1,
        SCMP_A0(SCMP_CMP_EQ, (scmp_datum_t)getpid())) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(prctl), 1,
        SCMP_A0(SCMP_CMP_EQ, PR_SET_NAME)) ||
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(prctl), 1,
        SCMP_A0(SCMP_CMP_EQ, PR_GET_NAME))) goto done;
    /* Only the harmless terminal queries used by FFmpeg; no arbitrary device
     * ioctl. All external socket, IPC, signal and filesystem mutation calls
     * (chmod/chown/xattr/utime/truncate etc.) remain denied. */
    if (seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(ioctl), 1,
        SCMP_A1(SCMP_CMP_EQ, 0x5401)) || /* TCGETS */
        seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(ioctl), 1,
        SCMP_A1(SCMP_CMP_EQ, 0x5413))) goto done; /* TIOCGWINSZ */
    if (seccomp_load(ctx)) goto done;
    result = 0;
done:
    seccomp_release(ctx);
    return result;
}

static int close_inherited_descriptors(void) {
    return syscall(SYS_close_range, 3U, ~0U, 0);
}

static int install_media_policy(const char *tool, const char *const *inputs,
                                size_t input_count, const char *const *outputs,
                                size_t output_count) {
    if (!approved_tool(tool) || landlock_policy(tool, inputs, input_count,
        outputs, output_count, NULL) || close_inherited_descriptors() ||
        syscall_policy(1)) return -1;
    return 0;
}

/* ctypes entrypoint for the already-running trusted offline ASR worker.
 * It receives its one decoded WAV and approved model directory, never the
 * artifact catalog, broker socket, or application credentials. */
int reskin_install_asr_policy(const char *audio_path, const char *model_dir,
                             int expected_parent_pid) {
    const char *inputs[] = {audio_path};
    /* Parent identity comes from the trusted LocalASR request, so a worker
     * orphaned even before entering this function cannot finish detached. */
    if (expected_parent_pid <= 0 || !audio_path || !model_dir ||
        audio_path[0] != '/' || model_dir[0] != '/' ||
        prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != expected_parent_pid ||
        landlock_policy(NULL, inputs, 1, NULL, 0, model_dir) ||
        close_inherited_descriptors() || syscall_policy(0)) return -1;
    return 0;
}

static int install_resource_limits(void) {
    struct {int resource; rlim_t value;} limits[] = {
        {RLIMIT_CORE, 0}, {RLIMIT_FSIZE, FILE_LIMIT}, {RLIMIT_NOFILE, 64},
        {RLIMIT_AS, 2ULL * 1024 * 1024 * 1024}, {RLIMIT_CPU, 1200}
    };
    for (size_t i = 0; i < sizeof(limits)/sizeof(limits[0]); ++i) {
        struct rlimit limit = {limits[i].value, limits[i].value};
        struct rlimit old;
        if (getrlimit(limits[i].resource, &old)) return -1;
        if (old.rlim_max < limit.rlim_max) limit.rlim_cur = limit.rlim_max = old.rlim_max;
        if (setrlimit(limits[i].resource, &limit)) return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *inputs[MAX_INPUTS], *outputs[MAX_OUTPUTS], *tool = NULL;
    size_t input_count = 0, output_count = 0;
    pid_t expected_parent = 0;
    int i = 1;
    for (; i < argc && strcmp(argv[i], "--"); i += 2) {
        if (i + 1 >= argc) goto fail;
        if (!strcmp(argv[i], "--tool") && !tool) tool = argv[i+1];
        else if (!strcmp(argv[i], "--parent-pid") && !expected_parent) {
            char *end = NULL;
            errno = 0;
            long value = strtol(argv[i+1], &end, 10);
            if (errno || !end || *end || value <= 0 || value > INT_MAX) goto fail;
            expected_parent = (pid_t)value;
        }
        else if (!strcmp(argv[i], "--input") && input_count < MAX_INPUTS) inputs[input_count++] = argv[i+1];
        else if (!strcmp(argv[i], "--output") && output_count < MAX_OUTPUTS) outputs[output_count++] = argv[i+1];
        else goto fail;
    }
    if (!approved_tool(tool) || !expected_parent || i + 1 >= argc || strcmp(argv[i], "--")) goto fail;
    /* Popen already supplies this allowlist. Repeat it here so tool startup
     * cannot inherit application/database/loader settings through this ABI. */
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd)) || clearenv() ||
        setenv("PATH", "/usr/bin:/bin", 1) || setenv("LANG", "C.UTF-8", 1) ||
        setenv("LC_ALL", "C.UTF-8", 1) || setenv("HOME", cwd, 1) ||
        setenv("TMPDIR", cwd, 1) || setenv("XDG_CACHE_HOME", cwd, 1) ||
        setenv("FONTCONFIG_FILE", "/etc/fonts/fonts.conf", 1) ||
        setenv("OMP_NUM_THREADS", "2", 1) || setenv("OPENBLAS_NUM_THREADS", "2", 1)) goto fail;
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != expected_parent ||
        install_resource_limits() || install_media_policy(tool, inputs,
        input_count, outputs, output_count)) goto fail;
    argv[i] = (char *)tool;
    execv(tool, &argv[i]);
fail:
    /* No path, argv, media bytes, or inherited environment in diagnostics. */
    fputs("media confinement unavailable\n", stderr);
    return 125;
}
