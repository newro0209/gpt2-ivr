"""GPT2-IVR 파이프라인 패키지.

Tokenizer Model Migration + IVR (Infrequent Vocabulary Replacement) 파이프라인을 제공한다.
BPE 토크나이저를 Unigram으로 증류하고, 저빈도 토큰을 고빈도 바이그램으로 교체하여
도메인 특화 토크나이저와 임베딩을 생성한다.
"""
