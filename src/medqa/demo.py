"""Gradio demo that talks to the vLLM OpenAI-compatible server started by `medqa serve`."""

from __future__ import annotations

from pathlib import Path

from medqa.config import ApiConfig, ExperimentConfig

DISCLAIMER = "Research demo. Not medical advice; do not use for clinical decisions."


def build_app(
    cfg: ExperimentConfig, api: ApiConfig, index_dir: Path | None = None, voice: bool = False
):
    import gradio as gr

    from medqa.inference.openai_engine import OpenAIGenerator

    generator = OpenAIGenerator(api, cfg.generation)
    retriever = None
    if index_dir is not None:
        from medqa.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever.load(index_dir)

    def answer(question: str, use_rag: bool = False) -> tuple[str, str]:
        contexts = None
        if use_rag and retriever is not None:
            contexts = [p.text for p in retriever.search([question], k=cfg.data.rag_top_k)[0]]
        text = generator.generate([question], [contexts] if contexts else None).texts[0]
        return text, "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts or [], start=1))

    with gr.Blocks(title="MedQA") as app:
        gr.Markdown(f"# MedQA: {cfg.model.display_name}\n{DISCLAIMER}")
        with gr.Tab("Ask"):
            question = gr.Textbox(
                label="Medical question",
                placeholder="What is the mechanism of action of metformin?",
            )
            use_rag = gr.Checkbox(
                label="Retrieve supporting flashcards (RAG)",
                value=False,
                visible=retriever is not None,
            )
            ask = gr.Button("Answer", variant="primary")
            reply = gr.Textbox(label="Answer", lines=4)
            sources = gr.Textbox(label="Retrieved context", lines=6, visible=retriever is not None)
            ask.click(answer, [question, use_rag], [reply, sources])
            question.submit(answer, [question, use_rag], [reply, sources])
            gr.Examples(
                [
                    "What is the first-line treatment for anaphylaxis?",
                    "Which nerve is damaged in wrist drop?",
                    "What does a positive Babinski sign indicate in adults?",
                ],
                inputs=question,
            )
        if voice:
            from medqa.voice.assistant import VoiceAssistant

            assistant = VoiceAssistant(lambda q: answer(q)[0])
            with gr.Tab("Voice"):
                mic = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Ask out loud")
                heard = gr.Textbox(label="Transcribed question")
                spoken = gr.Textbox(label="Answer")
                audio_out = gr.Audio(label="Spoken answer", autoplay=True)
                gr.Button("Answer").click(assistant.respond, mic, [heard, spoken, audio_out])
    return app
