from models.llm.qwen import QwenLLM


def main():
    model = QwenLLM(
        max_new_tokens=32,
        do_sample=False,
    )

    try:
        print("Loading Qwen...")

        model.load()

        print("Model loaded.")
        print()

        response = model.generate(
            prompt="Explain what machine learning is in one short paragraph.",
            sample_id="manual_qwen_001",
        )

        print("Model response:")
        print(response.text)
        print()

        print("Model name:", response.model_name)
        print("Model type:", response.model_type)
        print("Sample ID:", response.sample_id)

    finally:
        print()
        print("Unloading model...")
        model.unload()
        print("Model unloaded.")


if __name__ == "__main__":
    main()