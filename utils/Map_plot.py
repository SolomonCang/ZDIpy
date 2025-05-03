#!/usr/bin/python3
#
# Plot the three spatial components of the magnetic field,
# Use polar projection display (polar view),
# Display latitude range from -30° to 90°, reference line interval is 30°,
# The reference line at the equator is drawn with a thick line, and a unified colorbar is used (on the right, with upper and lower triangle tips),
# The colorbar label is $B$[G],
# And short lines indicating the observed phases are placed on the outer ring of the polar coordinates.
#
# In addition, adjust the distance between the colorbar and the figure,
# Only use 4 angular labels (0, 0.25, 0.50, 0.75),
# Set the starting position of 0° in polar coordinates at the bottom, and set the angular direction to clockwise.
# Also, set the colorbar to display discrete changes,
# Use discrete_levels divisions in both positive and negative directions (i.e., divide the interval from -BMax to BMax into 2*discrete_levels segments),
# And hope the colors near zero are white.
#

import numpy as np
try:
    import core.geometryStellar as geometryStellar
    import core.magneticGeom as magneticGeom
except ImportError:
    # If running from the utils subdirectory, try adding ZDIpy main directory to the path
    import sys
    sys.path += [sys.path[0] + '/../']
    import core.geometryStellar as geometryStellar
    import core.magneticGeom as magneticGeom


def main(magCoeffFile='outMagCoeff.dat', incDeg=20., discrete_levels=10):
    """
    Optional magnetic field spherical harmonic coefficient file,
    and the inclination of the star (kept for interface compatibility with original Hammer projection),
    
    Parameters:
      discrete_levels: Number of discrete levels for both positive and negative directions (default 6, i.e. 12 intervals)
    """
    # Observation parameters (original code logic retained)
    phaseList = [0.]

    # Number of latitudinal grid points for the spherical star surface
    nGridLatSph = 90

    # Number of latitude grid points used for cartesian magnetic mapping
    nGridLatRec = 90
    # Total number of grid points is nGridLatRec*(2*nGridLatRec)

    # Initialize star grid
    sGrid = geometryStellar.starGrid(nGridLatSph)

    # Initialize magnetic field spherical harmonic geometry from file----coefficient initialization
    magGeom = magneticGeom.magSphHarmoicsFromFile(magCoeffFile)

    plotPolarViewPhaseLat(magGeom,
                          nGridLatRec,
                          inclination=90.,
                          discrete_levels=discrete_levels)
    # Here inclination no longer affects the displayed region in polar coordinates


##########################################################
def plotPolarViewPhaseLat(inMagHarm,
                          npClat,
                          inclination=90.,
                          discrete_levels=15):
    """
    Plot the three magnetic field components using polar projection:
      - Radial
      - Azimuthal
      - Meridional
    
    Polar coordinate settings:
      - Radial coordinate r is converted from latitude: r = 90 - lat°,
        i.e., North Pole (lat=90°) corresponds to r=0, Equator (lat=0) corresponds to r=90,
        Lowest latitude (lat=-30°) corresponds to r=120.
      - Radial ticks every 30 degrees (tick locations: 0, 30, 60, 90, 120),
        and the equator (r=90, lat=0) is drawn with a thick line.
      - Angular direction (theta) uses the original longitude (phase) data.
    A unified colorbar is used and placed on the right side of the plot, with arrows at both ends, and the label is $B$[G].
    
    Other settings:
      - Adjust the distance between the colorbar and the figure;
      - Only use 4 angular labels (phase values: 0, 0.25, 0.50, 0.75);
      - Start polar coordinate 0° at the bottom, angular direction is clockwise;
      - Colorbar display is set to discrete levels,
        Using discrete_levels divisions for both positive and negative directions (a total of 2*discrete_levels intervals, corresponding to 2*discrete_levels+1 boundaries),
        And set the two intervals near zero to be white.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import cm, colors
    import numpy.ma as ma

    # Ensure using Type 3 font when exporting vector graphics
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    eps = 1e-5  # Avoid extreme value problems in polar coordinate conversion

    # Read observed phases (only used for adding reference markers in the plot)
    specname = './outLineModels.dat'
    cycle = []
    nspec = 0
    with open(specname, 'r') as inSpec:
        for line in inSpec:
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                # Assume the second number in each line is the phase
                cycle += [float(line.split()[1])]
                nspec += 1
    obsphase = np.array(cycle) % 1.

    # Set longitude and latitude grids
    npLon = 2 * npClat
    lon = np.linspace(0., 2 * np.pi, 2 * npClat)
    # Original clat array (unit: radian), range eps to pi-eps
    clat = np.linspace(eps, np.pi - eps, npClat)
    # Build full grid data: each clat corresponds to a full set of longitudes
    fullClat = np.repeat(clat, npLon)
    fullLon = np.tile(lon, npClat)

    # Convert to latitude: fullLat = 0.5*pi - fullClat
    fullLat = 0.5 * np.pi - fullClat
    # Longitude is flipped by 180° for plotting
    fullLonPlt = np.pi - np.tile(lon, npClat)

    # -----------------------------
    # Convert latitude to degrees, and calculate polar radial coordinate r
    lat_deg = fullLat * 180. / np.pi  # Convert to degrees
    # Define polar radial variable: r = 90 - lat_deg
    # lat = 90 -> r = 0, lat = 0 -> r = 90, lat = -30 -> r = 120
    r_vals = 90 - lat_deg

    # Build mask: only keep data where latitude >= -30°
    mask = (lat_deg < -30).reshape((npClat, npLon))
    # -----------------------------

    # Get magnetic field vector components
    magHarm = magneticGeom.magSphHarmoics(inMagHarm.nl)
    magHarm.alpha = inMagHarm.alpha
    magHarm.beta = inMagHarm.beta
    magHarm.gamma = inMagHarm.gamma
    magHarm.initMagGeom(fullClat, fullLon)
    Br_long, Bclat_long, Blon_long = magHarm.getAllMagVectors()
    # Reshape into 2D array and flip longitude direction
    Br = np.reshape(Br_long, (npClat, npLon))[:, ::-1]
    Blat = np.reshape(Bclat_long, (npClat, npLon))[:, ::-1]
    Blon = np.reshape(Blon_long, (npClat, npLon))[:, ::-1]
    # Reverse meridional component (so + direction corresponds to northern hemisphere)
    Blat = -Blat

    # Automatically determine color scale range (all three components share one color scale, and range is symmetric)
    BrMax = np.amax(np.abs(Br))
    BlatMax = np.amax(np.abs(Blat))
    BlonMax = np.amax(np.abs(Blon))
    BMax = np.amax([BrMax, BlatMax, BlonMax])

    # Define discrete level boundaries: total intervals = 2*discrete_levels, so number of boundaries = 2*discrete_levels+1
    levels = np.linspace(-BMax, BMax, 2 * discrete_levels + 1)

    # Get discrete colors for RdBu_r, here divided into 2*discrete_levels segments
    base_cmap = cm.get_cmap('RdBu_r')
    discrete_colors = base_cmap(np.linspace(0, 1, 2 * discrete_levels))
    # If the number of discrete colors is even (e.g. discrete_levels = 6, then 12 colors), the middle two indices are discrete_levels-1 and discrete_levels
    discrete_colors[discrete_levels - 1] = (1.0, 1.0, 1.0, 1.0)
    discrete_colors[discrete_levels] = (1.0, 1.0, 1.0, 1.0)
    new_cmap = colors.ListedColormap(discrete_colors)

    # Construct discrete normalization mapping
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=new_cmap.N)

    # Mask data: exclude regions with latitude below -30°
    BrMask = ma.masked_where(mask, Br)
    BlonMask = ma.masked_where(mask, Blon)
    BlatMask = ma.masked_where(mask, Blat)

    # Only use 4 angular labels (phase values: 0, 0.25, 0.50, 0.75)
    labelPhases = [0.0, 0.25, 0.50, 0.75]

    # Create figure, set size to (8, 10)
    fig = plt.figure(figsize=(10, 5))

    # Unified polar radial range: r from 0 to 120
    rlim = (0, 120)
    # Radial ticks (corresponding to lat=90, 60, 30, 0, -30)
    rticks = [0, 30, 60, 90, 120]
    rtick_labels = ["90°", "60°", "30°", "0°", "-30°"]
    rtick_labels = []

    # Observed phase tick marks: place short lines on the outermost ring, extending tick_length units
    tick_length = 8
    tick_outer = rlim[1] - tick_length
    tick_outer_end = rlim[1]

    #-------------------- Plot radial field --------------------
    ax1 = plt.subplot(1, 3, 1, projection="polar")
    ax1.set_theta_zero_location('S')  # Set 0° at the bottom
    ax1.set_theta_direction(-1)  # Angular direction clockwise
    mesh1 = ax1.pcolormesh(fullLonPlt.reshape((npClat, npLon)),
                           r_vals.reshape((npClat, npLon)),
                           BrMask,
                           cmap=new_cmap,
                           norm=norm,
                           shading='auto')
    ax1.set_ylim(rlim)
    ax1.set_yticks(rticks)
    ax1.set_yticklabels(rtick_labels)
    ax1.set_title('Radial', loc='center')
    ax1.set_xticks(np.linspace(0, 2 * np.pi, len(labelPhases), endpoint=False))
    ax1.set_xticklabels(['{:3.2f}'.format(x) for x in labelPhases])
    ax1.grid(True)
    for phase_val in obsphase:
        theta = phase_val * 2 * np.pi
        ax1.plot([theta, theta], [tick_outer, tick_outer_end],
                 'k',
                 linewidth=1)
    theta_circle = np.linspace(0, 2 * np.pi, 360)
    ax1.plot(theta_circle,
             np.full_like(theta_circle, 90),
             color='k',
             linewidth=2)

    #-------------------- Plot azimuthal field --------------------
    ax2 = plt.subplot(1, 3, 2, projection="polar")
    ax2.set_theta_zero_location('S')
    ax2.set_theta_direction(-1)
    mesh2 = ax2.pcolormesh(fullLonPlt.reshape((npClat, npLon)),
                           r_vals.reshape((npClat, npLon)),
                           BlonMask,
                           cmap=new_cmap,
                           norm=norm,
                           shading='auto')
    ax2.set_ylim(rlim)
    ax2.set_yticks(rticks)
    ax2.set_yticklabels(rtick_labels)
    ax2.set_title('Azimuthal', loc='center')
    ax2.set_xticks(np.linspace(0, 2 * np.pi, len(labelPhases), endpoint=False))
    ax2.set_xticklabels(['{:3.2f}'.format(x) for x in labelPhases])
    ax2.grid(True)
    for phase_val in obsphase:
        theta = phase_val * 2 * np.pi
        ax2.plot([theta, theta], [tick_outer, tick_outer_end],
                 'k',
                 linewidth=1)
    ax2.plot(theta_circle,
             np.full_like(theta_circle, 90),
             color='k',
             linewidth=2)

    #-------------------- Plot meridional field --------------------
    ax3 = plt.subplot(1, 3, 3, projection="polar")
    ax3.set_theta_zero_location('S')
    ax3.set_theta_direction(-1)
    mesh3 = ax3.pcolormesh(fullLonPlt.reshape((npClat, npLon)),
                           r_vals.reshape((npClat, npLon)),
                           BlatMask,
                           cmap=new_cmap,
                           norm=norm,
                           shading='auto')
    ax3.set_ylim(rlim)
    ax3.set_yticks(rticks)
    ax3.set_yticklabels(rtick_labels)
    ax3.set_title('Meridional', loc='center')
    ax3.set_xticks(np.linspace(0, 2 * np.pi, len(labelPhases), endpoint=False))
    ax3.set_xticklabels(['{:3.2f}'.format(x) for x in labelPhases])
    ax3.grid(True)
    for phase_val in obsphase:
        theta = phase_val * 2 * np.pi
        ax3.plot([theta, theta], [tick_outer, tick_outer_end],
                 'k',
                 linewidth=1)
    ax3.plot(theta_circle,
             np.full_like(theta_circle, 90),
             color='k',
             linewidth=2)

    plt.subplots_adjust(left=0.05, wspace=0.3)
    cax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(mesh1,
                        cax=cax,
                        orientation='horizontal',
                        boundaries=levels,
                        ticks=np.linspace(-BMax, BMax, 4),
                        extend='both')
    cbar.set_label(r"$B$[G]", fontsize=12)

    plt.show()
    fig.savefig('Magmap_polar.png')


##################################################################
if __name__ == "__main__":
    main()
