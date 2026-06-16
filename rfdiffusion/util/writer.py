
import torch

from rfdiffusion.rosettafold.chemical import num2aa, aa2long

# writepdb
def writepdb(
    filename, atoms, seq, binderlen=None, idx_pdb=None, bfacts=None, chain_idx=None
):
    f = open(filename, "w")
    ctr = 1
    scpu = seq.cpu().squeeze()
    atomscpu = atoms.cpu().squeeze()
    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    Bfacts = torch.clamp(bfacts.cpu(), 0, 1)
    for i, s in enumerate(scpu):
        if chain_idx is None:
            if binderlen is not None:
                if i < binderlen:
                    chain = "A"
                else:
                    chain = "B"
            elif binderlen is None:
                chain = "A"
        else:
            chain = chain_idx[i]
        if len(atomscpu.shape) == 2:
            f.write(
                "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"
                % (
                    "ATOM",
                    ctr,
                    " CA ",
                    num2aa[s],
                    chain,
                    idx_pdb[i],
                    atomscpu[i, 0],
                    atomscpu[i, 1],
                    atomscpu[i, 2],
                    1.0,
                    Bfacts[i],
                )
            )
            ctr += 1
        elif atomscpu.shape[1] == 3:
            for j, atm_j in enumerate([" N  ", " CA ", " C  "]):
                f.write(
                    "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"
                    % (
                        "ATOM",
                        ctr,
                        atm_j,
                        num2aa[s],
                        chain,
                        idx_pdb[i],
                        atomscpu[i, j, 0],
                        atomscpu[i, j, 1],
                        atomscpu[i, j, 2],
                        1.0,
                        Bfacts[i],
                    )
                )
                ctr += 1
        elif atomscpu.shape[1] == 4:
            for j, atm_j in enumerate([" N  ", " CA ", " C  ", " O  "]):
                f.write(
                    "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"
                    % (
                        "ATOM",
                        ctr,
                        atm_j,
                        num2aa[s],
                        chain,
                        idx_pdb[i],
                        atomscpu[i, j, 0],
                        atomscpu[i, j, 1],
                        atomscpu[i, j, 2],
                        1.0,
                        Bfacts[i],
                    )
                )
                ctr += 1

        else:
            natoms = atomscpu.shape[1]
            if natoms != 14 and natoms != 27:
                print("bad size!", atoms.shape)
                assert False
            atms = aa2long[s]
            # his prot hack
            if (
                s == 8
                and torch.linalg.norm(atomscpu[i, 9, :] - atomscpu[i, 5, :]) < 1.7
            ):
                atms = (
                    " N  ",
                    " CA ",
                    " C  ",
                    " O  ",
                    " CB ",
                    " CG ",
                    " NE2",
                    " CD2",
                    " CE1",
                    " ND1",
                    None,
                    None,
                    None,
                    None,
                    " H  ",
                    " HA ",
                    "1HB ",
                    "2HB ",
                    " HD2",
                    " HE1",
                    " HD1",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )  # his_d

            for j, atm_j in enumerate(atms):
                if (
                    j < natoms and atm_j is not None
                ):  # and not torch.isnan(atomscpu[i,j,:]).any()):
                    f.write(
                        "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"
                        % (
                            "ATOM",
                            ctr,
                            atm_j,
                            num2aa[s],
                            chain,
                            idx_pdb[i],
                            atomscpu[i, j, 0],
                            atomscpu[i, j, 1],
                            atomscpu[i, j, 2],
                            1.0,
                            Bfacts[i],
                        )
                    )
                    ctr += 1

N_BACKBONE_ATOMS = 3
N_HEAVY = 14


def writepdb_multi(
    filename,
    atoms_stack,
    bfacts,
    seq_stack,
    backbone_only=False,
    chain_ids=None,
    use_hydrogens=True,
):
    """
    Function for writing multiple structural states of the same sequence into a single
    pdb file.
    """

    f = open(filename, "w")

    if seq_stack.ndim != 2:
        T = atoms_stack.shape[0]
        seq_stack = torch.tile(seq_stack, (T, 1))
    seq_stack = seq_stack.cpu()
    for atoms, scpu in zip(atoms_stack, seq_stack):
        ctr = 1
        atomscpu = atoms.cpu()
        Bfacts = torch.clamp(bfacts.cpu(), 0, 1)
        for i, s in enumerate(scpu):
            atms = aa2long[s]
            for j, atm_j in enumerate(atms):
                if backbone_only and j >= N_BACKBONE_ATOMS:
                    break
                if not use_hydrogens and j >= N_HEAVY:
                    break
                if (atm_j is None) or (torch.all(torch.isnan(atomscpu[i, j]))):
                    continue
                chain_id = "A"
                if chain_ids is not None:
                    chain_id = chain_ids[i]
                f.write(
                    "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"
                    % (
                        "ATOM",
                        ctr,
                        atm_j,
                        num2aa[s],
                        chain_id,
                        i + 1,
                        atomscpu[i, j, 0],
                        atomscpu[i, j, 1],
                        atomscpu[i, j, 2],
                        1.0,
                        Bfacts[i],
                    )
                )
                ctr += 1

        f.write("ENDMDL\n")
