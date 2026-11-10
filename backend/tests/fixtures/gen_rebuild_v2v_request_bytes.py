"""Record the exact v2v createTask request bytes main builds (run inside the image, cwd=backend)."""
import json, sys
import httpx, respx
sys.path.insert(0, ".")
from app import ai_models, kie_client, pipeline_v2

CREATE = "https://api.kie.ai/api/v1/jobs/createTask"
out = {}
for model in ai_models.AI_MODELS:
    for ga in (None, False, True):
        recipe = dict(
            run_model=model, effective_prompt="swap the character",
            ref_urls=["https://x/r1.png", "https://x/r2.png"], clip_url="https://x/clip.mp4",
            clip_start=12.3, clip_end=19.65, run_resolution="720p",
            project_aspect_ratio="9:16", project_width=1080, project_height=1920,
            generate_audio=ga,
            audio_ref_urls=(["https://x/v.mp3"] if model == "seedance-2-5" and ga else None),
        )
        try:
            method, kwargs = pipeline_v2._swap_task_call(**recipe)
        except Exception as exc:  # noqa: BLE001
            print("skip", model, ga, exc, file=sys.stderr)
            continue
        with respx.mock(assert_all_called=False) as m:
            route = m.post(url__regex=r".*").mock(return_value=httpx.Response(
                200, json={"code": 200, "data": {"taskId": "t1"}}))
            getattr(kie_client.KieClient(api_key="k", sleep_fn=lambda s: None), method)(**kwargs)
        req = route.calls[0].request
        out[f"{model}|{ga}"] = {"method": method, "url": str(req.url),
                                "body": req.content.decode("utf-8")}
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(len(out), "payloads")
