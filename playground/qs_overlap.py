import numpy as np
import quicksilver as qs

space = qs.CISpace(norbital=4, na=2, nb=2)

stringsa = space.stringsa
stringsb = space.stringsb

print(stringsa)

a = "1001"
idxa = stringsa.index(int(a, 2))

statevector = qs.SparseElectronicStatevector()
statevector[0, 0] = 1.0
statevector[idxa, 0] = -0.42
print(statevector[0, 0])
print(statevector)
print(statevector.pretty_str(space))

statevector1 = np.zeros(
    (
        space.nstringa,
        space.nstringb,
    ),
    dtype=np.float64,
)
qs.SparseElectronicStatevectorUtility.prepare_statevector(
    space=space,
    sparse_statevector=statevector,
    statevector=statevector1,
    dtype=np.float64,
)


statevector2_sp = qs.SparseElectronicStatevector()
statevector2_sp[idxa, 0] = 1

statevector2 = np.zeros(
    (
        space.nstringa,
        space.nstringb,
    ),
    dtype=np.float64,
)
qs.SparseElectronicStatevectorUtility.prepare_statevector(
    space=space,
    sparse_statevector=statevector2_sp,
    statevector=statevector2,
    dtype=np.float64,
)

overlap = np.vdot(statevector2.conj(), statevector1)
