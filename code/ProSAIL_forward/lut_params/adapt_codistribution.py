import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_lut = pd.read_csv('prosail_danner-etal_europe_soil.csv')
df_cod = pd.read_csv('codistribution_snap.csv')

# Define LAI ranges
lai_min_snap, lai_max_snap = 0.0, 15.0  # For SNAP
lai_min_data, lai_max_data = 0.0, 10.0  # For my data
lai_snap = np.linspace(lai_min_snap, lai_max_snap, 200)
lai_data = np.linspace(lai_min_data, lai_max_data, 200)

# Keep only LUT rows for parameters in codistribution
cod_params = df_cod["param"].values
df_lut_sub = df_lut[df_lut["Parameter"].isin(cod_params)]

n_params = len(df_lut_sub)

fig, axes = plt.subplots(
    n_params, 3,
    figsize=(11, 3.2 * n_params),
    constrained_layout=True
)

# Ensure axes is always 2D
if n_params == 1:
    axes = np.array([axes])

def generate_distribution(row, n=10000):
    dist_type = row["Distribution"]
    vmin = row["Min"]
    vmax = row["Max"]
    mode = row["Mode"]
    std = row["Std"]

    if dist_type.lower() == "gaussian":
        samples = np.random.normal(loc=mode, scale=std, size=n)
        samples = np.clip(samples, vmin, vmax)

    elif dist_type.lower() == "uniform":
        samples = np.random.uniform(vmin, vmax, size=n)

    elif dist_type.lower() == "constant":
        samples = np.full(n, mode)

    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

    return samples

# Create a list to store the adapted parameter ranges
adapted_params = []

for i, (_, lut_row) in enumerate(df_lut_sub.iterrows()):
    param = lut_row["Parameter"]

    # ---- LEFT COLUMN: CODISTRIBUTION SPACE ----
    cod_row = df_cod[df_cod["param"] == param].iloc[0]

    Vmin0 = cod_row["Vmin0"]
    Vmax0 = cod_row["Vmax0"]
    VminLAI_snap = cod_row["Vmin(LAImax)"]
    VmaxLAI_snap = cod_row["Vmax(LAImax)"]

    lower_snap = Vmin0 + (VminLAI_snap - Vmin0) * (lai_snap / lai_max_snap)
    upper_snap = Vmax0 + (VmaxLAI_snap - Vmax0) * (lai_snap / lai_max_snap)

    ax_cod = axes[i, 0]
    ax_cod.fill_between(lai_snap, lower_snap, upper_snap, alpha=0.4)
    ax_cod.plot(lai_snap, lower_snap)
    ax_cod.plot(lai_snap, upper_snap)

    ax_cod.set_title(f" SNAP Codistribution space: {param}")
    ax_cod.set_xlabel("LAI")
    ax_cod.set_ylabel(param)
    ax_cod.set_xlim(lai_min_snap, lai_max_snap)
    
    ax_cod.axhline(lut_row["Min"], linestyle="--")
    ax_cod.axhline(lut_row["Max"], linestyle="--")
    ax_cod.axhline(lut_row["Mode"], linestyle=":")

    # ---- ACTUAL PARAMETER DISTRIBUTION ----
    samples = generate_distribution(lut_row)

    ax_dist = axes[i, 1]
    ax_dist.hist(samples, bins=60, density=True)
    ax_dist.axvline(lut_row["Min"], linestyle="--")
    ax_dist.axvline(lut_row["Max"], linestyle="--")
    ax_dist.axvline(lut_row["Mode"], linestyle=":")

    ax_dist.set_title(
        f"{param} distribution\n({lut_row['Distribution']})"
    )
    ax_dist.set_xlabel(param)
    ax_dist.set_ylabel("Density")


    # ---- ADJUSTED SNAP CODISTRIBUTION ----

    new_bounds = {
      'cab': [20,80,45,80],
      'cbrown': [0,1,0,1],
      'cm': [0.003,0.01,0.005,0.01],
      'cw': [0,0.07,0,0.07],
      'lidfa':[30,70,55,65]
    }

    if param not in new_bounds.keys():
      cod_row = df_cod[df_cod["param"] == param].iloc[0]
      Vmin0 = cod_row["Vmin0"]
      Vmax0 = cod_row["Vmax0"]
      VminLAI = cod_row["Vmin(LAImax)"]
      VmaxLAI = cod_row["Vmax(LAImax)"]
    else:
      cod_row = new_bounds[param]
      Vmin0 = cod_row[0]
      Vmax0 = cod_row[1]
      VminLAI = cod_row[2]
      VmaxLAI = cod_row[3]

    # Extract VminLAI and VmaxLAI at LAI=10 based on SNAP
    VminLAI_data = Vmin0 + (VminLAI_snap - Vmin0) * (lai_max_data / lai_max_snap)
    VmaxLAI_data = Vmax0 + (VmaxLAI_snap - Vmax0) * (lai_max_data / lai_max_snap)
    
    # Ensure VminLAI_data and VmaxLAI_data are within lut_row["Min"] and lut_row["Max"]
    if VminLAI_data > lut_row["Max"]:
        VminLAI_data = lut_row["Min"]
    else:
        VminLAI_data = np.clip(VminLAI_data, lut_row["Min"], lut_row["Max"])
    VmaxLAI_data = np.clip(VmaxLAI_data, lut_row["Min"], lut_row["Max"])

    # Print the parameter and its limits for debugging
    print(f"Parameter: {param}, VminLAI_data: {VminLAI_data:.2f}, VmaxLAI_data: {VmaxLAI_data:.2f}, "
        f"lut_row Min: {lut_row['Min']}, lut_row Max: {lut_row['Max']}")

    # Calculate lower and upper bounds for your data
    lower_data = Vmin0 + (VminLAI_data - Vmin0) * (lai_data / lai_max_data)
    upper_data = Vmax0 + (VmaxLAI_data - Vmax0) * (lai_data / lai_max_data)

    ax_cod2 = axes[i, 2]
    ax_cod2.fill_between(lai_data, lower_data, upper_data, alpha=0.4)
    ax_cod2.plot(lai_data, lower_data)
    ax_cod2.plot(lai_data, upper_data)
    ymin, ymax = ax_cod.get_ylim()
    ax_cod2.set_ylim(ymin, ymax)
    ax_cod2.set_xlim(lai_min_data, lai_max_data)

    ax_cod2.set_title(f"Adjusted Codistribution space: {param}")
    ax_cod2.set_xlabel("LAI")
    ax_cod2.set_ylabel(param)
    
    ax_cod2.axhline(lut_row["Min"], linestyle="--")
    ax_cod2.axhline(lut_row["Max"], linestyle="--")
    ax_cod2.axhline(lut_row["Mode"], linestyle=":")

    # Append the adapted parameter ranges to the list
    adapted_params.append({
        "param": param,
        "Vmin0": Vmin0,
        "Vmax0": Vmax0,
        "Vmin(LAImax)": VminLAI_data,
        "Vmax(LAImax)": VmaxLAI_data
    })

plt.savefig('codistribution_spaces_europesoil.png')


# Save the adapted parameters to a new CSV file
df_adapted = pd.DataFrame(adapted_params)
df_adapted = df_adapted.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
df_adapted.to_csv('codistribution_adapted.csv', index=False)