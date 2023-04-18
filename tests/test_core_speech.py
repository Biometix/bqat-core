from bqat_core.speech import process_speech
import json
import math


def test_speech_normal():
    """
    GIVEN a set of mock speech samples
    WHEN the samples processed by the module
    THEN check the output values are within the expected range
    """
    # Case 1
    with open("data/conformance/speech/tests.json") as f:
        tests = [json.loads(line) for line in f.readlines()]

    for test in tests:
        test.pop("_id")
        test.pop("tag")

        output = process_speech(test.pop("file"), "file")

        assert isinstance(output, dict)
        assert output.get("log") == None
        for k, _ in test.items():
            assert math.isclose(float(output[k]), float(test[k]), abs_tol=0.001)
