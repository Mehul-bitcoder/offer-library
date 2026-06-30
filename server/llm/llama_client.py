from pathlib import Path
import yaml
from llama_cpp import Llama


class LlamaClient:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.llm = self._load_model()

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["llm"]

    def _load_model(self) -> Llama:
        # Resolve model path relative to project root (parent of config file's directory)
        config_file = Path(self.config_path).resolve()
        project_root = config_file.parent.parent  # configs -> project root
        model_path = project_root / self.config["model_path"]

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path.absolute()}")

        print("Loading model...")
        try:
            llm = Llama(
                model_path=str(model_path),
                n_ctx=self.config.get("n_ctx", 4096),
                n_threads=self.config.get("n_threads", 8),
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                verbose=True,
            )
            print("Model loaded successfully.")
            return llm
        except AssertionError as e:
            print(f"\nERROR: Model initialization failed - likely unsupported architecture")
            print(f"This typically means llama-cpp-python doesn't support this model type.")
            print(f"\nModel path: {model_path}")
            print(f"Model exists: {model_path.exists()}")
            print(f"Model size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Model path: {model_path}")
            print(f"Model exists: {model_path.exists()}")
            print(f"Model size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes")
            raise

    def generate(self, prompt: str) -> str:
        response = self.llm(
            prompt,
            max_tokens=self.config.get("max_tokens", 512),
            stop=["</s>"],
        )

        return response["choices"][0]["text"].strip()


# -------- POC TEST --------
if __name__ == "__main__":
    client = LlamaClient(config_path="configs/model.yaml")

    test_prompt = (
        "You are a helpful assistant.\n"
        "Question: What is a vector database?\n"
        "Answer:"
    )

    output = client.generate(test_prompt)
    print("\n--- MODEL OUTPUT ---")
    print(output)
