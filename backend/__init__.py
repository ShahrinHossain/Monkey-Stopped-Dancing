def __init__(self) -> None:
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        self._backend_name = "deep_translator"
        self._backend = GoogleTranslator
        print("Initialized deep-translator backend.")
        return
    except Exception as e:
        print(f"Failed to initialize deep-translator: {e}")

    try:
        from googletrans import Translator as GoogleTransTranslator  # type: ignore
        self._backend_name = "googletrans"
        self._backend = GoogleTransTranslator()
        print("Initialized googletrans backend.")
        return
    except Exception as e:
        print(f"Failed to initialize googletrans: {e}")

    self._backend_name = None
    self._backend = None
    print("No translation backend available.")