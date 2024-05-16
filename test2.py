
import numpy as np
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        df = data
        df["Date"] = df["Date"].dt.strftime("%d-%b-%Y")
        return df

    def define_test_input(test_case_id):
        if test_case_id == 1:
            df = pd.DataFrame(
                {"Date": ["2019-01-01", "2019-02-08", "2019-02-08", "2019-03-08"]}
            )
            df["Date"] = pd.to_datetime(df["Date"])
        return df

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0


exec_context = r"""
import pandas as pd
import numpy as np
df = test_input
[insert]
result = df
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
