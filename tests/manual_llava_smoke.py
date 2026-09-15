from pathlib import Path

from models.vlm.llava import LLaVAVLM


def main():
    image_path = Path(
        "tests/fixtures/vlm_test_image.png"
    )

    model = LLaVAVLM(
        max_new_tokens=32,
        do_sample=False,
    )

    try:
        print("Loading LLaVA-1.5-7B...")
        model.load()

        print("Model loaded.")
        print()

        response = model.generate(
            prompt=(
                "Describe this image "
                "in one short sentence."
            ),
            image_path=image_path,
            sample_id="manual_llava_001",
        )

        print("Model response:")
        print(response.text)
        print()

        print("Model name:", response.model_name)
        print("Model type:", response.model_type)
        print("Sample ID:", response.sample_id)
        print("Image:", response.image_path)

    finally:
        print()
        print("Unloading model...")

        model.unload()

        print("Model unloaded.")


if __name__ == "__main__":
    main()