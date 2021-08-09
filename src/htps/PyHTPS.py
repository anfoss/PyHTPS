import pandas as pd
import math as m
import numpy as np
from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def get_aa_distr():
    aa = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "W",
        "Y",
        "K",
        "V",
    ]
    val = [
        104162,
        21798,
        75974,
        114861,
        47771,
        92794,
        31054,
        65094,
        130814,
        32829,
        52269,
        80896,
        68808,
        78356,
        106859,
        74448,
        13544,
        36490,
        96627,
        86690,
    ]
    perc = [
        7.376191279,
        1.543616842,
        5.380069087,
        8.133836778,
        3.382884676,
        6.571170806,
        2.199076861,
        4.609606143,
        9.263542232,
        2.324772791,
        3.701408786,
        5.728618591,
        4.8726116,
        5.548749485,
        7.567178279,
        5.272005994,
        0.959113061,
        2.584025074,
        6.842603202,
        6.138918434,
    ]
    return pd.DataFrame({"AA": aa, "value": val, "perc": perc})


def count_aa_distr(sq):
    '''Count aa distribution in a fasta file

    Args:
    sq: sequence list from parse fasta

    Returns:
    dataframe with AA, value and percentage
    '''
    pass


def parse_fasta(db):
    '''Parse a fasta file

    Args:
    db: fasta filename

    Returns:
    two list one for ids and one for sequence
    '''
    from Bio import SeqIO

    sq, id = [], []
    for record in SeqIO.parse(db, "fasta"):
        sq.append(str(record.seq))
        id.append(record.id.split("|")[1])
    return id, sq


def match_peptide(id2seq, pid, pepseq, q):
    '''Matches a peptide within a sequence and return the cleavage window +-q

    Args:
    id2seq: dictionary mapping id to peptide seq (need to have same id)
    pid: protein ids
    pepseq: sequence to search
    q : window size (default 8)

    Returns:
    q sized cleavage windows as a list
    '''
    # idx of cleavage site
    idx = id2seq[pid].find(pepseq) + len(pepseq)
    # if cleavage is C terminali
    if idx == len(id2seq[pid]):
        seq = [np.nan] * q*2
    elif len(id2seq[pid]) - idx < q:
        # then center cleavage and use pad the cterm with nan
        # q aa
        nterm = list(id2seq[pid][idx - q : idx])
        cterm = list(id2seq[pid][idx:])
        while len(cterm) <q:
            cterm.append(np.nan)
        nterm.extend(cterm)
        seq = nterm
    # if cleavage is N term
    elif idx - q < 0:
        # q aa at the c term
        cterm = list(id2seq[pid][idx: idx+q])
        nterm = list(id2seq[pid][:idx])
        while len(nterm) <q:
            nterm.insert(0, np.nan)
        seq = nterm.extend(cterm)
        seq = nterm
    else:
        seq = list(id2seq[pid][idx - q : idx + q])

    return pd.Series(seq, dtype="object")


def convert_to_cleavage(filename, db, search="MQ", qc=False, q=8):
    '''Receive a search file and returns a formatted cleavage matrix

    Args:
    filename: txt/tsv file generated from a search engine
    search: flag signalling the search engine used
    db: dict mapping ids to sequence
    qc: bool, qc filtering for high scoring psm
    q: cleavage window size (int)

    Returns:
    formatted cleavage matrix
    '''
    df = pd.read_table(filename, sep="\t")
    if search == "MQ":
        df = df[df["Intensity"] > 0]
        if qc:
            df = df[(df["Score"] > 40) & (df["PEP"] <= 0.05)]
        df = df[["Sequence", "Leading razor protein"]]
        df = df[df["Leading razor protein"].str.contains("sp")]
        df["ID"] = [x.split("|")[1] for x in list(df["Leading razor protein"])]
        df = df.drop("Leading razor protein", axis=1)
        df.columns = ["Sequence", "Protein ID"]
    elif search == "Fragger":
        df = df[["Peptide", "Protein ID"]]
        df.columns = ["Sequence", "Protein ID"]
    elif search == "custom":
        pass
    df.drop_duplicates(subset=["Sequence"], inplace=True)
    df = df.apply(
        (lambda x: match_peptide(db, x["Protein ID"], x["Sequence"], q)), axis=1
    )
    w = len(list(df)) // 2
    df.columns = [str(x) for x in range(1, w + 1)][::-1] + [str(x)+'\'' for x in range(1, w + 1)]
    return df


def expand_counts(df):
    '''Receive a cleavage matrix and add missing aa with 0 counts per position

    Args:
    df: cleavage matrix

    Returns:
    filled cleavage matrix
    '''

    aa = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    # rmeove - nan or no aa
    df = df.loc[df.index.isin(aa)]
    for k in aa:
        if k not in df.index:
            row = dict(zip(list(df.columns), [0] * len(list(df.columns))))
            # df = df.append(pd.Series(row, index=df.columns, name=k))
            df = df.append(row, ignore_index=True)
    return df


def entropy(arr):
    '''Calculates shannon entropy per position

    Args:
    arr: a position (column) from a cleavage matrix

    Returns:
    one entropy value per position
    '''
    import math as m

    d = [m.log(x, 20) * x for x in list(arr.values)]
    entr = -1 * np.nansum(np.array(d))
    return entr


def subpocket_e(entr_df):

    v = entr_df.apply(lambda col: entropy(col), axis=0)
    v = pd.Series([(x - min(v)) / (max(v) - min(v)) for x in v])
    return v


def blocks_e(entropy_arr):
    '''Calculates subpocket entropy per position according to
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003007
    local entropy by summing block wise the entropies
    i.e p4 to p1 available
    4 blocks
    b4 = entropy(p4,p3,p2,p1)
    b3 = entropy(p3,p2,p1)
    b2 = entropy (p2,p1)
    b1 = entropy (p1)

    Args:
    entropy_arr: entropy series

    Returns:
    block entropy per position
    '''
    tmp = list(entropy_arr)
    w = len(tmp) // 2
    left = [sum(tmp[x:w]) for x in range(0, w)]
    right = [sum(tmp[w:x]) for x in range(w + 1, len(tmp) + 1)]
    return pd.Series(left + right)


def sequence_decoy(htps_db, df, n=50):
    '''generate a cleavage matrix by random sampling peptides in htps DB
    Args:
    htps db: list of aa from string concatenated htps db
    df: cleavage matrix (peptide  level)
    n: number of iterations

    Returns:
    normalized frequency matrix
    '''
    # generate random cleavage matrix N times
    mtrx = []
    nrow, ncol = df.shape
    seed = 111
    for x in range(1, n):
        np.random.seed(seed)
        seed += 1
        arr = np.empty(df.shape, dtype="object")
        for i in range(0, ncol):
            arr[:, i] = np.random.choice(htps_db, nrow)
        dec_df = pd.DataFrame(arr)
        dec_df = dec_df.apply(pd.value_counts, result_type="expand")
        dec_df = expand_counts(dec_df)
        mtrx.append(dec_df)
    # average frequency across all matrixes
    base_df = mtrx.pop()
    for m1 in mtrx:
        base_df = np.add(base_df, m1)
    base_df = base_df / (len(mtrx) + 1)
    return base_df


def normalize_aa_abundance(df, aa):
    '''normalize a cleavage frequency matrix by the frequency of aa in htps db fasta
    Args:
    df: cleavage frequency matrix

    Returns:
    normalized frequency matrix
    '''
    df = df.apply((lambda x: x / aa["value"]), axis=0)
    df = df.apply((lambda x: x / np.sum(x)), axis=0)
    return df


def aa_importance(target, decoy):
    '''train RF model and extract feature importance per position

    target = cleavage matrix
    decoy = decoy cleavage matrix
    '''
    target['class'] = 1
    decoy['class'] = 0
    cmb = pd.concat([target, decoy])
    # conc = cmb.values.tolist()).str.join('')
    # print(conc)
    # assert False
    y = cmb['class'].values
    X = cmb.drop(['class'], axis=1)
    aa = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    char_dict = {}
    for index, val in enumerate(aa):
        char_dict[val] = index+1
    # one hot encode
    X = X.replace(char_dict)
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X, y)
    print(rf_clf.feature_importances_)
    assert False

    # split
    df = pd.DataFrame(
    {'feature': list(target),
     'importance': rf_clf.feature_importances_})

    # filter features without importance
    df = df[df['importance']>0]
    df.sort_values(['importance'],ascending=False, inplace=True)

    # keep first 20
    top20 = df.iloc[0:20]
    sns.barplot(x="importance", y="feature", data=top20, color="salmon", saturation=.5)
    plt.show()
    df.to_csv('test.csv')
    assert False
    # train


    # extract features

    # return feature map


def convert_cleavage_matrx(cleav_mtrx, aa_distr, pref, write=True):
    '''process a single cleavage matrix into its specificity, entropy and block entropy
    Args:
    cleav_mtrx: cleavage frequency matrix (n aa, n sites)
    pref: string prefix to be added to the specificity, cleavage and block entropy df
    aa : dataframe with distribution of aa
    write: bool, write output to file

    Returns:
    spec_df = cleavage specificity dataframe per position (aa, n sites)
    entr_df = entropy per position (n sites)
    block_df = block entropy per position (n sites)
    '''
    # sort to ensure same order between all matrixes
    cleav_mtrx.sort_index(inplace=True)
    spec_df = cleav_mtrx.apply((lambda x: x / np.sum(x)), axis=0)
    spec_df = np.log2(spec_df)
    cleav_mtrx = normalize_aa_abundance(cleav_mtrx, aa_distr)
    entr_mtrx = cleav_mtrx.apply(entropy, axis=0)
    bl_entr_mtrx = blocks_e(entr_mtrx)
    entr_mtrx.name = "{} Entropy".format(pref)
    bl_entr_mtrx.name = "{} Block entropy".format(pref)
    if write:
        cleav_mtrx.to_csv("{}_cleavage_prob.csv".format(pref))
        entr_mtrx.to_csv("{}_entropy.csv".format(pref))
        bl_entr_mtrx.to_csv("{}_block_entropy.csv".format(pref))
        spec_df.to_csv("{}_specificity.csv".format(pref))
    return spec_df, cleav_mtrx, entr_mtrx, bl_entr_mtrx


def permutation_pvalue(target, decoy, p=0.1, i=500):
    '''Permutation p value per position per AA

    target: dataframe of specificity AA
    decoy: dataframe of decoy specificity
    p = p value threshold
    i: number of iteration
    '''
    from functools import reduce
    diff = np.abs(target.values) - np.abs(decoy.values)
    v = np.hstack((target.values, decoy.values)).flatten()
    rnd = []
    for x in range(0,i):
        np.random.seed(x)
        t_rnd = np.random.choice(v, size=(target.shape))
        d_rnd = np.random.choice(v, size=(decoy.shape))
        np.random.shuffle(v)
        diff_rnd = (np.abs(diff) > np.abs(t_rnd - d_rnd)).astype(int)
        rnd.append(diff_rnd)

    base = rnd.pop()
    for x in rnd:
        np.add(base, x)
    base = base/i
    base[base>p] = 0
    pval = pd.DataFrame(base, columns=target.columns, index=target.index)
    return pval


def plot_spec_matrix(matrix, output):
    '''Permutation p value per position per AA

    target: dataframe of specificity AA
    decoy: dataframe of decoy specificity
    p = p value threshold
    i: number of iteration
    '''
    # sns.color_palette("viridis", as_cmap=True)
    fig, ax = plt.subplots(figsize=(5,5))
    g = sns.heatmap(
        matrix,
        cmap="coolwarm",
        #square=True,
        cbar_kws={"shrink": .82},
        linewidths=0.1,
        linecolor='black',
        ax=ax
    )
    for _, spine in g.spines.items():
        spine.set_visible(True)
    g.set(xlabel="Position", ylabel="AA", aspect="equal")
    plt.ylabel("AA", fontsize=9)
    plt.xlabel("Position", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    plt.savefig('{}.pdf'.format(output), bbox_inches='tight', dpi=1600)
    plt.close()


def main():
    '''
    process a single file
    '''
    ids, seq = parse_fasta("HTPS_db.fasta")
    cleav_mtrx = convert_to_cleavage(
        "peptides.txt", dict(zip(ids, seq)), search="MQ", qc=False, q=10
    )
    db = list("".join(seq))
    dec_cleav_mtrx = sequence_decoy(db, cleav_mtrx, 100)
    # aa_imp = aa_importance(cleav_mtrx, dec_cleav_mtrx)
    # assert False
    cleav_mtrx = cleav_mtrx.apply(pd.value_counts, result_type="expand")
    cleav_mtrx = expand_counts(cleav_mtrx)
    aa_distr = get_aa_distr().set_index("AA")
    spec_df, cleav_mtrx, entr_mtrx, bl_entr_mtrx = convert_cleavage_matrx(
        cleav_mtrx, aa_distr, "Furin", True
    )
    spec_dec, cleav_dec, entr_dec, bl_entr_dec = convert_cleavage_matrx(
        dec_cleav_mtrx, aa_distr, "", False
    )
    dff = spec_df.values - spec_dec.values
    dff = pd.DataFrame(dff, index=spec_df.index)
    w = len(list(dff)) // 2
    # convert it to the form p-1 and p 1' for p prime
    dff.columns = [-x for x in range(1, w + 1)][::-1] + [x for x in range(1, w + 1)]
    dff.to_csv("delta_spec{}.csv".format("Furin"))
    pval_df = permutation_pvalue(spec_df, spec_dec, p=0.05, i=1000)
    plot_spec_matrix(dff, 'furin_fc')
    dff[pval_df.values == 0] = np.nan
    # dff.to_csv("delta_spec{}_filtered.csv".format("Furin"))
    plot_spec_matrix(dff, 'furin_fc_filtered')



if __name__ == "__main__":
    main()
