from bqat_core.face import scan_face
import json


def test_face_default():
    """
    GIVEN a set of mock face images and default engine
    WHEN the images processed by the module
    THEN check the output values are within the expected range
    """
    with open("data/conformance/face/tests_default.json") as f:
        tests = [json.loads(line) for line in f.readlines()]

    for test in tests:
        test.pop("_id")
        test.pop("tag")

        output = scan_face(test.pop("file"), engine="default")

        assert isinstance(output, dict)
        assert output.get("log") == None
        for k, _ in test.items():
            assert output[k] == test[k]


def test_face_biqt():
    """
    GIVEN a set of mock face images and alternate engine
    WHEN the images processed by the module
    THEN check the output values are within the expected range
    """
    with open("data/conformance/face/tests_biqt.json") as f:
        tests = [json.loads(line) for line in f.readlines()]

    for test in tests:
        test.pop("_id")
        test.pop("tag")

        output = scan_face(test.pop("file"), engine="biqt")

        assert isinstance(output, dict)
        assert output.get("log") == None
        for k, _ in test.items():
            assert output[k] == test[k]
