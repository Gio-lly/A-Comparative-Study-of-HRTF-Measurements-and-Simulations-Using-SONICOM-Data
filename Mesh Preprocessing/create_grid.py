import netCDF4 as nc
import numpy as np

sofa_path = "/home/gio/Documents/SONICOM SOFAs/HRIR_SONICOM_44100.sofa"

with nc.Dataset(sofa_path, 'r') as ds:
    # 1. Extract positions (grid coordinates)
    # Standard SOFA variable for the measurement points
    source_pos = ds.variables['SourcePosition'][:]

    # 2. Attempt to retrieve the coordinate type
    # Try to find it as a variable attribute or a global one
    try:
        coord_type = ds.variables['SourcePosition'].getncattr('Type')
    except AttributeError:
        try:
            coord_type = ds.getncattr('SourcePositionType')
        except AttributeError:
            coord_type = "Non specificato (assumo sferico)"

    # 3. Retrieve the unit of measurement
    try:
        units = ds.variables['SourcePosition'].getncattr('Units')
    except AttributeError:
        units = "degree, degree, metre"

print(f"--- Info Griglia ---")
print(f"Tipo Coordinate: {coord_type}")
print(f"Unità: {units}")
print(f"Formato Array: {source_pos.shape}") # Should be (NumberOfPoints, 3)

# 4. Save for Mesh2HRTF
# Mesh2HRTF expects a clean .txt file with X Y Z (or Az El R)
# If the units contain 'degree', the coordinates are spherical
if 'degree' in units.lower():
    print("Rilevate coordinate sferiche. Conversione in corso...")
    
    def sph2cart(az_deg, el_deg, r):
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    cartesian_points = [sph2cart(p[0], p[1], p[2]) for p in source_pos]
    final_grid = np.array(cartesian_points)
else:
    final_grid = source_pos

# Save the final file that Mesh2HRTF will use as EvaluationGrid
np.savetxt("EvaluationGrid.txt", final_grid, fmt='%.6f')
print("Fatto! File 'EvaluationGrid.txt' generato.")