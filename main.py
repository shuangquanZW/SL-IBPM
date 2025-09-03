# from ibpm import main as slibpm
from wo_aggr import main as without_aggr
from wo_damp import main as without_damp
from wo_lambda import main as without_lambda


if __name__ == "__main__":
    # slibpm()
    without_aggr()
    without_damp()
    without_lambda()
