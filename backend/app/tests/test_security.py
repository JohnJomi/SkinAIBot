from app.core.security import create_access_token, decode_access_token, hash_password, verify_password


def test_hash_password_verifies_correct_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", hashed)


def test_hash_password_rejects_incorrect_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert not verify_password("wrong-password", hashed)


def test_access_token_roundtrip():
    token = create_access_token(subject="user-123")
    assert decode_access_token(token) == "user-123"


def test_decode_access_token_rejects_garbage_token():
    assert decode_access_token("not-a-real-token") is None
