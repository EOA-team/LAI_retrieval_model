"""
Python implementation of SNAP LAI 

Selene Ledain, 01 Dec 2025
"""
import numpy as np


deg_to_rad = np.pi / 180

def normalize(unnormalized, min_val, max_val):
    return 2 * (unnormalized - min_val) / (max_val - min_val) - 1

def denormalize(normalized, min_val, max_val):
    return 0.5 * (normalized + 1) * (max_val - min_val) + min_val

def tansig(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


# ===== Layer 1: Neurons =====
def neuron1(b03,b04,b05,b06,b07,b8a,b11,b12,viewZen,sunZen,relAzim):
    s = (
        + 4.96238030555279
        - 0.023406878966470 * b03
        + 0.921655164636366 * b04
        + 0.135576544080099 * b05
        - 1.938331472397950 * b06
        - 3.342495816122680 * b07
        + 0.902277648009576 * b8a
        + 0.205363538258614 * b11
        - 0.040607844721716 * b12
        - 0.083196409727092 * viewZen
        + 0.260029270773809 * sunZen
        + 0.284761567218845 * relAzim
    )
    return tansig(s)


def neuron2(b03,b04,b05,b06,b07,b8a,b11,b12,viewZen,sunZen,relAzim):
    s = (
        + 1.416008443981500
        - 0.132555480856684 * b03
        - 0.139574837333540 * b04
        - 1.014606016898920 * b05
        - 1.330890038649270 * b06
        + 0.031730624503341 * b07
        - 1.433583541317050 * b8a
        - 0.959637898574699 * b11
        + 1.133115706551000 * b12
        + 0.216603876541632 * viewZen
        + 0.410652303762839 * sunZen
        + 0.064760155543506 * relAzim
    )
    return tansig(s)


def neuron3(b03,b04,b05,b06,b07,b8a,b11,b12,viewZen,sunZen,relAzim):
    s = (
        + 1.075897047213310
        + 0.086015977724868 * b03
        + 0.616648776881434 * b04
        + 0.678003876446556 * b05
        + 0.141102398644968 * b06
        - 0.096682206883546 * b07
        - 1.128832638862200 * b8a
        + 0.302189102741375 * b11
        + 0.434494937299725 * b12
        - 0.021903699490589 * viewZen
        - 0.228492476802263 * sunZen
        - 0.039460537589826 * relAzim
    )
    return tansig(s)


def neuron4(b03,b04,b05,b06,b07,b8a,b11,b12,viewZen,sunZen,relAzim):
    s = (
        + 1.533988264655420
        - 0.109366593670404 * b03
        - 0.071046262972729 * b04
        + 0.064582411478320 * b05
        + 2.906325236823160 * b06
        - 0.673873108979163 * b07
        - 3.838051868280840 * b8a
        + 1.695979344531530 * b11
        + 0.046950296081713 * b12
        - 0.049709652688365 * viewZen
        + 0.021829545430994 * sunZen
        + 0.057483827104091 * relAzim
    )
    return tansig(s)


def neuron5(b03,b04,b05,b06,b07,b8a,b11,b12,viewZen,sunZen,relAzim):
    s = (
        + 3.024115930757230
        - 0.089939416159969 * b03
        + 0.175395483106147 * b04
        - 0.081847329172620 * b05
        + 2.219895367487790 * b06
        + 1.713873975136850 * b07
        + 0.713069186099534 * b8a
        + 0.138970813499201 * b11
        - 0.060771761518025 * b12
        + 0.124263341255473 * viewZen
        + 0.210086140404351 * sunZen
        - 0.183878138700341 * relAzim
    )
    return tansig(s)


# ===== Layer 2 (output) =====
def layer2(n1, n2, n3, n4, n5):
    return (
        + 1.096963107077220
        - 1.500135489728730 * n1
        - 0.096283269121503 * n2
        - 0.194935930577094 * n3
        - 0.352305895755591 * n4
        + 0.075107415847473 * n5
    )


# ===== Main prediction function =====
def predict_lai(X):
    """
    Predict LAI from a numpy array.
    
    Expected column order:
    [B03, B04, B05, B06, B07, B8A, B11, B12, viewZen, sunZen, relAzim]
    
    X shape: (n_samples, 11)
    """

    # --- Unpack bands ---
    b03 = X[:, 1] 
    b04 = X[:, 2]
    b05 = X[:, 3]
    b06 = X[:, 4]
    b07 = X[:, 5]
    b8a = X[:, 6]
    b11 = X[:, 7]
    b12 = X[:, 8]
    viewZen = X[:, 9]
    sunZen  = X[:, 10]
    relAzim = X[:, 11]

    # --- Normalize bands ---
    b03 = normalize(b03, 0, 0.253061520471542)
    b04 = normalize(b04, 0, 0.290393577911328)
    b05 = normalize(b05, 0, 0.305398915248555)
    b06 = normalize(b06, 0.006637972542253, 0.608900395797889)
    b07 = normalize(b07, 0.013972727018939, 0.753827384322927)
    b8a = normalize(b8a, 0.026690138082061, 0.782011770669178)
    b11 = normalize(b11, 0.016388074192258, 0.493761397883092)
    b12 = normalize(b12, 0, 0.493025984460231)

    viewZen = normalize(np.cos(viewZen * deg_to_rad), 0.918595400582046, 1)
    sunZen  = normalize(np.cos(sunZen  * deg_to_rad), 0.342022871159208, 0.936206429175402)
    relAzim = np.cos(relAzim * deg_to_rad)

    # --- Compute neurons ---
    n1 = neuron1(b03, b04, b05, b06, b07, b8a, b11, b12, viewZen, sunZen, relAzim)
    n2 = neuron2(b03, b04, b05, b06, b07, b8a, b11, b12, viewZen, sunZen, relAzim)
    n3 = neuron3(b03, b04, b05, b06, b07, b8a, b11, b12, viewZen, sunZen, relAzim)
    n4 = neuron4(b03, b04, b05, b06, b07, b8a, b11, b12, viewZen, sunZen, relAzim)
    n5 = neuron5(b03, b04, b05, b06, b07, b8a, b11, b12, viewZen, sunZen, relAzim)

    l2 = layer2(n1, n2, n3, n4, n5)

    lai = denormalize(l2, 0.000319182538301, 14.4675094548151)

    return lai


def predict_snap_df(df):
    """
    Predict LAI from a DataFrame with S2 band reflectances and acquisition angles.

    Required columns: B03, B04, B05, B06, B07, B8A, B11, B12,
                      mean_sensor_zenith, mean_solar_zenith, relative_azimuth
    """
    band_cols = ['B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12',
                 'mean_sensor_zenith', 'mean_solar_zenith', 'relative_azimuth']
    X = np.column_stack([np.zeros(len(df)), df[band_cols].values])
    return predict_lai(X)