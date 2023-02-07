from bqat_core.finger import scan_finger
import csv
import random


def test_finger_nfiq2():
    """
    GIVEN the nfiq2 conformance dataset
    WHEN the images processed by the module
    THEN check if the output values conform
    """
    # Case 1
    with open("data/conformance/finger/expected.csv") as f:
        tests = list(csv.DictReader(f))
        tests = random.sample(tests, 10)

    for test in tests:
        file = "data/conformance/finger/images/" + test["Filename"]

        output = scan_finger(file)

        assert isinstance(output, dict)
        if output.get("log"):
            assert output["log"]["nfiq2"] == test["OptionalError"]
        else:
            assert output["NFIQ2"] == test["QualityScore"]
