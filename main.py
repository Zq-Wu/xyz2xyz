import numpy as np
element_dict = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73
}
def read_xyz(filename):
    with open(filename) as f:
        lines = f.readlines()
    num_atoms = int(lines[0])
    comment = lines[1].strip()
    coords = []
    symbols = []
    for line in lines[2:]:
        symbol, x, y, z = line.split()
        symbols.append(symbol)
        coords.append([float(x), float(y), float(z)])
    return symbols, np.array(coords), comment

def write_xyz(filename, symbols, coords, comment):
    with open(filename, 'w') as f:
        f.write(f"{comment}\n")
        for i in range(len(symbols)):
            x, y, z = coords[i]
            num = element_dict[symbols[i]]
            f.write(f"{num:<2.0f}   {x: .6f}   {y: .6f}   {z: .6f}    1.00000\n")
        f.write("-1\n")

def convert_xyz(input_filename, output_filename):
    symbols, coords, comment = read_xyz(input_filename)
    write_xyz(output_filename, symbols, coords, comment)

convert_xyz("input.xyz", "output.xyz")