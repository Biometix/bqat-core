from bqat_core.iris import scan_iris
import json


def test_iris_normal():
    """
    GIVEN a set of mock iris images
    WHEN the images processed by the module
    THEN check the output values are within the expected range
    """
    # Case 1
    with open("data/conformance/iris/tests.json") as f:
        tests = [json.loads(line) for line in f.readlines()]

    for test in tests:
        test.pop("_id")
        test.pop("tag")

        output = scan_iris(test.pop("file"))

        assert isinstance(output, dict)
        assert output.get("log") == None
        for k, _ in test.items():
            assert output[k] == test[k]
