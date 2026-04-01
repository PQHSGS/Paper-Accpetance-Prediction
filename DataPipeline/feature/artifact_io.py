import operator

def read_features(ifile: str) -> dict:
    idToFeature = dict()
    with open(ifile, "rb") as ifh:
        for l in ifh:
            e = l.rstrip().decode("utf-8").split("\t")
            if len(e) == 2:
                idToFeature[e[1]] = int(e[0])
    return idToFeature

def save_features_to_file(idToFeature: dict, feature_output_file: str):
    """
    Saves features dictionary to TSV tab separated format.
    """
    with open(feature_output_file, 'wb') as ofh:
        sorted_items = sorted(list(idToFeature.items()), key=operator.itemgetter(1))
        for i in sorted_items:
            # i[1] is the ID, i[0] is the feature string name based on dictionary sort
            line_str = f"{i[1]}\t{i[0]}\n".encode("utf-8")
            ofh.write(line_str)
            
def write_svmlite_row(label: int, feature_dict: dict, file_handle):
    """
    Writes a single sparse format row directly to a file handle.
    """
    file_handle.write(str(label) + " ")
    for k in sorted(feature_dict.keys()):
        file_handle.write(f"{k}:{feature_dict[k]} ")
    file_handle.write("\n")
