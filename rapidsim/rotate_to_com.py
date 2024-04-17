import uproot
import pandas as pd
import numpy as np
import argh
import vector


def srq_check(px, py, pz, E, m):
    print(px, py, pz, E, m)

    vx, vy, vz, t = px/m, py/m, pz/m, E/m

    gamma = 1. / (np.sqrt(1 - (vx**2 + vy**2 + vz**2)))

    Ltt = gamma


    # Lta = Lat = lambda i: -v

def main(infile='data/single_mode_train.root', outfile='data/single_mode_train_rot.root'):

    print("Attempting eta load")
    file = uproot.open(infile)["DecayTree"]
    keys = file.keys()

    print(keys)

    results_np = file.arrays(keys, library="np")


    results = pd.DataFrame.from_dict(results_np)
    mother_P = np.sqrt(results.mother_PX ** 2 + results.mother_PY ** 2 + results.mother_PZ ** 2)
    mother_P_true = np.sqrt(
        results.mother_PX_TRUE ** 2 + results.mother_PY_TRUE ** 2 + results.mother_PZ_TRUE ** 2)

    pe_1 = np.sqrt(
        results.particle_1_M ** 2 + results.particle_1_PX ** 2 + results.particle_1_PY ** 2 + results.particle_1_PZ ** 2)
    pe_2 = np.sqrt(
        results.particle_2_M ** 2 + results.particle_2_PX ** 2 + results.particle_2_PY ** 2 + results.particle_2_PZ ** 2)
    pe_3 = np.sqrt(
        results.particle_3_M ** 2 + results.particle_3_PX ** 2 + results.particle_3_PY ** 2 + results.particle_3_PZ ** 2)

    pe = pe_1 + pe_2 + pe_3
    px = results.particle_1_PX + results.particle_2_PX + results.particle_3_PX
    py = results.particle_1_PY + results.particle_2_PY + results.particle_3_PY
    pz = results.particle_1_PZ + results.particle_2_PZ + results.particle_3_PZ

    B = vector.array({"px": px.to_numpy(), "py": py.to_numpy(), "pz": pz.to_numpy(), "E": pe.to_numpy()})

    P1 = vector.array(
        {"px": results.particle_1_PX, "py": results.particle_1_PY, "pz": results.particle_1_PZ, "E": pe_1})
    P2 = vector.array(
        {"px": results.particle_2_PX, "py": results.particle_2_PY, "pz": results.particle_2_PZ, "E": pe_2})
    P3 = vector.array(
        {"px": results.particle_3_PX, "py": results.particle_3_PY, "pz": results.particle_3_PZ, "E": pe_3})

    P1_ROT = P1.boostCM_of(B)
    P2_ROT = P2.boostCM_of(B)
    P3_ROT = P3.boostCM_of(B)

    PM_ROT = B.boostCM_of(B)



    pe_1_true = np.sqrt(
        results.particle_1_M_TRUE ** 2 + results.particle_1_PX_TRUE ** 2 + results.particle_1_PY_TRUE ** 2 + results.particle_1_PZ_TRUE ** 2)
    pe_2_true = np.sqrt(
        results.particle_2_M_TRUE ** 2 + results.particle_2_PX_TRUE ** 2 + results.particle_2_PY_TRUE ** 2 + results.particle_2_PZ_TRUE ** 2)
    pe_3_true = np.sqrt(
        results.particle_3_M_TRUE ** 2 + results.particle_3_PX_TRUE ** 2 + results.particle_3_PY_TRUE ** 2 + results.particle_3_PZ_TRUE ** 2)

    pe_true = pe_1_true + pe_2_true + pe_3_true
    px_true = results.particle_1_PX_TRUE + results.particle_2_PX_TRUE + results.particle_3_PX_TRUE
    py_true = results.particle_1_PY_TRUE + results.particle_2_PY_TRUE + results.particle_3_PY_TRUE
    pz_true = results.particle_1_PZ_TRUE + results.particle_2_PZ_TRUE + results.particle_3_PZ_TRUE


    B_true = vector.array({"px": px_true.to_numpy(), "py": py_true.to_numpy(), "pz": pz_true.to_numpy(), "E": pe_true.to_numpy()})

    P1_true = vector.array(
        {"px": results.particle_1_PX_TRUE, "py": results.particle_1_PY_TRUE, "pz": results.particle_1_PZ_TRUE, "E": pe_1_true})
    P2_true = vector.array(
        {"px": results.particle_2_PX_TRUE, "py": results.particle_2_PY_TRUE, "pz": results.particle_2_PZ_TRUE, "E": pe_2_true})
    P3_true = vector.array(
        {"px": results.particle_3_PX_TRUE, "py": results.particle_3_PY_TRUE, "pz": results.particle_3_PZ_TRUE, "E": pe_3_true})


    P1_ROT_TRUE = P1_true.boostCM_of(B_true)
    P2_ROT_TRUE = P2_true.boostCM_of(B_true)
    P3_ROT_TRUE = P3_true.boostCM_of(B_true)

    PM_ROT_TRUE = B_true.boostCM_of(B_true)


    print(type(B_true[0].px))

    srq_check(B_true[0].px, B_true[0].py, B_true[0].pz, B_true[0].E, B_true[0].mass)


    print(PM_ROT.mass)
    print(P1_ROT.mass)
    print(P2_ROT.mass)
    print(P3_ROT.mass)

    print(PM_ROT_TRUE.mass)
    print(P1_ROT_TRUE.mass)
    print(P2_ROT_TRUE.mass)
    print(P3_ROT_TRUE.mass)

    outdata = {'nEvent': results_np['nEvent'],
               'mother_PX': PM_ROT.px,
               'mother_PX_TRUE': PM_ROT_TRUE.px,
               'mother_PY': PM_ROT.py,
               'mother_PY_TRUE': PM_ROT_TRUE.py,
               'mother_PZ': PM_ROT.pz,
               'mother_PZ_TRUE': PM_ROT_TRUE.pz,
               'mother_E': PM_ROT.E,
               'mother_E_TRUE': PM_ROT_TRUE.E,
               'mother_M': PM_ROT.mass,
               'mother_M_TRUE': PM_ROT_TRUE.mass,
               'particle_1_PX': P1_ROT.px,
               'particle_1_PX_TRUE': P1_ROT_TRUE.px,
               'particle_2_PX': P2_ROT.px,
               'particle_2_PX_TRUE': P2_ROT_TRUE.px,
               'particle_3_PX': P3_ROT.px,
               'particle_3_PX_TRUE': P3_ROT_TRUE.px,
               'particle_1_PY': P1_ROT.py,
               'particle_1_PY_TRUE': P1_ROT_TRUE.py,
               'particle_2_PY': P2_ROT.py,
               'particle_2_PY_TRUE':  P2_ROT_TRUE.py,
               'particle_3_PY': P3_ROT.py,
               'particle_3_PY_TRUE':  P3_ROT_TRUE.py,
               'particle_1_PZ': P1_ROT.pz,
               'particle_1_PZ_TRUE':  P1_ROT_TRUE.pz,
               'particle_2_PZ': P2_ROT.pz,
               'particle_2_PZ_TRUE': P2_ROT_TRUE.pz,
               'particle_3_PZ': P3_ROT.pz,
               'particle_3_PZ_TRUE': P3_ROT_TRUE.pz,
               'particle_1_E': P1_ROT.E,
               'particle_1_E_TRUE': P1_ROT_TRUE.E,
               'particle_2_E': P2_ROT.E,
               'particle_2_E_TRUE': P2_ROT_TRUE.E,
               'particle_3_E': P3_ROT.E,
               'particle_3_E_TRUE': P3_ROT_TRUE.E,
               'particle_1_M': P1_ROT.mass,
               'particle_1_M_TRUE': P1_ROT_TRUE.mass,
               'particle_2_M': P2_ROT.mass,
               'particle_2_M_TRUE': P2_ROT_TRUE.mass,
               'particle_3_M': P3_ROT.mass,
               'particle_3_M_TRUE': P3_ROT_TRUE.mass,
               'mother_PID': results_np['mother_PID'],
               'particle_1_PID': results_np['particle_1_PID'],
               'particle_2_PID': results_np['particle_2_PID'],
               'particle_3_PID': results_np['particle_3_PID'],
               }

    # results = pd.DataFrame.from_dict(outdata)
    # mother_P = np.sqrt(results.mother_PX ** 2 + results.mother_PY ** 2 + results.mother_PZ ** 2)
    # mother_P_true = np.sqrt(
    #     results.mother_PX_TRUE ** 2 + results.mother_PY_TRUE ** 2 + results.mother_PZ_TRUE ** 2)
    #
    # pe_1 = np.sqrt(
    #     results.particle_1_M ** 2 + results.particle_1_PX ** 2 + results.particle_1_PY ** 2 + results.particle_1_PZ ** 2)
    # pe_2 = np.sqrt(
    #     results.particle_2_M ** 2 + results.particle_2_PX ** 2 + results.particle_2_PY ** 2 + results.particle_2_PZ ** 2)
    # pe_3 = np.sqrt(
    #     results.particle_3_M ** 2 + results.particle_3_PX ** 2 + results.particle_3_PY ** 2 + results.particle_3_PZ ** 2)
    #
    # pe = pe_1 + pe_2 + pe_3
    # px = results.particle_1_PX + results.particle_2_PX + results.particle_3_PX
    # py = results.particle_1_PY + results.particle_2_PY + results.particle_3_PY
    # pz = results.particle_1_PZ + results.particle_2_PZ + results.particle_3_PZ
    #
    # B = vector.array({"px": px.to_numpy(), "py": py.to_numpy(), "pz": pz.to_numpy(), "E": pe.to_numpy()})
    #
    # P1 = vector.array(
    #     {"px": results.particle_1_PX, "py": results.particle_1_PY, "pz": results.particle_1_PZ, "E": pe_1})
    # P2 = vector.array(
    #     {"px": results.particle_2_PX, "py": results.particle_2_PY, "pz": results.particle_2_PZ, "E": pe_2})
    # P3 = vector.array(
    #     {"px": results.particle_3_PX, "py": results.particle_3_PY, "pz": results.particle_3_PZ, "E": pe_3})
    #
    # # P1_ROT = P1.boostCM_of(B)
    # # P2_ROT = P2.boostCM_of(B)
    # # P3_ROT = P3.boostCM_of(B)
    # #
    # # PM_ROT = B.boostCM_of(B)
    #
    # print(P3.mass)
    #
    # 0/0

    file2 = uproot.recreate(outfile)
    file2['DecayTree'] = outdata
    file2.close()


if __name__=='__main__':
    argh.dispatch_command(main(infile='data/D+B+_1mode.root', outfile='data/D+B+_1mode_rot.root'))