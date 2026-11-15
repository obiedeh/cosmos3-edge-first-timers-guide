"""Send the Part 17 requests to a running vLLM server and print what comes back.

Part 17 of the notebook describes how the notebook's parameters map onto a
vLLM server. Those cells are reference text, because vLLM needs its own
environment. This script is the evidence behind that text: it sends each
request the section talks about and prints the token counts, the reply and
any error, so every figure in the section can be traced to a line of output.

Run it from a folder that holds `sample.png`, `digits.mp4` and
`long_digits.mp4` (the notebook writes all three), against a server started
as shown at the top of the saved output files:

    python run_server_tests.py http://127.0.0.1:8011/v1
"""

import base64
import sys

from openai import OpenAI

MODEL = "nvidia/Cosmos3-Edge"
SYSTEM = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}
COLOR = "What color is the tape? One word."
DIGITS = (
    "Large white digits flash on the screen one at a time during this video. "
    "List every digit that appears, in order, separated by commas. Answer with the digits only."
)
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def data_url(path: str, mime: str) -> str:
    """Return a file as a base64 data URL, the form the chat API accepts for media."""
    with open(path, "rb") as stream:
        return f"data:{mime};base64," + base64.b64encode(stream.read()).decode()


def call(client: OpenAI, name: str, media: dict, text: str, max_tokens: int = 60, **extra: object) -> None:
    """Send one request with greedy decoding and print its token counts and reply, or its error."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[SYSTEM, {"role": "user", "content": [media, {"type": "text", "text": text}]}],
            max_tokens=max_tokens,
            temperature=0,
            extra_body=extra or None,
        )
    except Exception as error:  # the error text is the result for the refused requests
        print(f"[{name}]\n    ERROR {type(error).__name__}: {str(error)[:260]}")
        return
    message = response.choices[0].message
    print(
        f"[{name}]\n    prompt_tokens={response.usage.prompt_tokens} "
        f"completion_tokens={response.usage.completion_tokens} finish={response.choices[0].finish_reason}\n"
        f"    content={message.content!r:.300}"
    )


def main() -> None:
    """Run every Part 17 request in a fixed order."""
    client = OpenAI(base_url=sys.argv[1], api_key="EMPTY")
    image = {"type": "image_url", "image_url": {"url": data_url("sample.png", "image/png")}}
    clip = {"type": "video_url", "video_url": {"url": data_url("digits.mp4", "video/mp4")}}
    long_clip = {"type": "video_url", "video_url": {"url": data_url("long_digits.mp4", "video/mp4")}}

    call(client, "image, nothing set (thinking is on by default)", image, COLOR, 300)
    call(client, "image, thinking off", image, COLOR, **THINKING_OFF)
    call(client, "video, server defaults, thinking off", clip, DIGITS, **THINKING_OFF)
    for fps in (1, 4):
        call(client, f"video, media_io_kwargs fps={fps}", clip, DIGITS,
             media_io_kwargs={"video": {"fps": fps, "num_frames": -1}}, **THINKING_OFF)
    call(client, "video, fps=4 inside mm_processor_kwargs", clip, DIGITS,
         mm_processor_kwargs={"fps": 4, "do_sample_frames": True}, **THINKING_OFF)
    call(client, "image, mm_processor_kwargs max_pixels = 256 x 1024", image, COLOR,
         mm_processor_kwargs={"max_pixels": 256 * 1024}, **THINKING_OFF)
    call(client, "video, fps=4 and max_pixels = 4096 x 1024", clip, DIGITS,
         media_io_kwargs={"video": {"fps": 4, "num_frames": -1}},
         mm_processor_kwargs={"max_pixels": 4096 * 1024}, **THINKING_OFF)
    call(client, "one-minute video, fps=4 (longer than the context limit)", long_clip, DIGITS,
         media_io_kwargs={"video": {"fps": 4, "num_frames": -1}}, **THINKING_OFF)
    call(client, "image, max_tokens above the context limit", image, COLOR, 9000, **THINKING_OFF)
    call(client, "image, top_k and repetition_penalty in extra_body", image,
         "In one sentence, what should the robot do next?", top_k=20, repetition_penalty=1.0, **THINKING_OFF)


if __name__ == "__main__":
    main()
