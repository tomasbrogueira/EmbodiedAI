"""Independent exclusion flags shared by numerical stages and display consumers."""
SEMANTIC=1
GEOMETRY=2
CLEARANCE=4
HEIGHT_CONFLICT=8
UNKNOWN_SLOPE=16
STEEP=32
RETRY=64
UNKNOWN_GROUND=128
LABELS={SEMANTIC:'semantic obstacle',GEOMETRY:'observed geometry',CLEARANCE:'insufficient clearance',
    HEIGHT_CONFLICT:'height disagreement',UNKNOWN_SLOPE:'unknown slope',STEEP:'steep ground',
    RETRY:'route validation failed',UNKNOWN_GROUND:'missing ground evidence'}

def counts(grid):
    import numpy as np
    visible=grid['point_rows']>=0;values=grid['reasons'].ravel()
    return {label:int(np.count_nonzero(visible&((values&flag)!=0))) for flag,label in LABELS.items()}
