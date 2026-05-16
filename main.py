# from ibpm import main as slibpm
# from ibpm import cascade as slibpm_cascade
from wo_aggr import main as without_aggr
from wo_damp import main as without_damp
from wo_lambda import main as without_lambda
from wo_damp_aggr import main as without_damp_aggr
from wo_epsilon import main as without_epsilon
from plain_gnn import main as plain_gnn_baseline

# from ajc import main as ajc
# from ddsml import main as ddsml
# from gcnsi import main as gcnsi
# from hfsd import main as hfsd
# from ivgd import main as ivgd, cascade_study as ivgd_cascade
# from lpsi import main as lpsi
# from mpnn import main as mpnn, cascade_study as mpnn_cascade
# from rdgin import main as rdgin, cascade_study as rdgin_cascade
# from slvae import main as slvae

if __name__ == "__main__":
    # print("=" * 60)
    # print("Running SL-IBPM (100 epochs)")
    # slibpm(epochs=100)

    # print("\nRunning SL-IBPM Cascade")
    # slibpm_cascade(epochs=100)

    print("Running Ablation Studies (10 epochs)")
    without_aggr(epochs=100)
    without_damp(epochs=100)
    without_lambda(epochs=100)
    # without_damp_aggr(epochs=100)
    # without_epsilon(epochs=100)
    # plain_gnn_baseline(epochs=100)

    # print("\nRunning Baseline")
    # ajc()
    # ddsml(epochs=100)
    # gcnsi()
    # hfsd()
    # ivgd()
    # ivgd_cascade()
    # lpsi()
    # mpnn()
    # mpnn_cascade()
    # rdgin()
    # rdgin_cascade()
    # slvae()
    # print("=" * 60)
    # print("\nAll experiments completed!")
