from ivgd import cascade_study as ivgd_cascade_study
from mpnn import cascade_study as mpnn_cascade_study
from rdgin import cascade_study as rdgin_cascade_study
from ibpm import cascade as ibpm_cascade


def cascade_study():
    rdgin_cascade_study()
    mpnn_cascade_study()
    ivgd_cascade_study()
    ibpm_cascade()
