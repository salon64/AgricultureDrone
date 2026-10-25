# Orchard layouts

Real-world tree positions that `scripts/orchard_env.py` loads when `LAYOUT_MODE = "file"`.
Each `*.csv` is one orchard block; the environment spawns tree actors at runtime
(`Tree_LowPoly` static mesh, see `airsim/import_tree_mesh.bat`) and, every episode,
places a random window of 4 to 8 trees from a random layout in front of the drone,
rotated by a random heading. No trees need to be placed in the Unreal editor any more.

## Files in this folder

| Layout | Where | What it is |
|---|---|---|
| `kinpo_kawachi_block.csv` | Kawachi, Nishi-ku, Kumamoto (32.8711 N, 130.6037 E) | Young mikan replant block, regular rows about 2.5 m apart |
| `kinpo_kawachi_east.csv` | same block plus surrounding mature trees | Mixed ages, in-row about 4.3 m, rows about 5.8 m |
| `kinpo_kawachi_west.csv` | Kawachi, south-west of the road (32.8709 N, 130.6018 E) | Mature mikan, dense canopy, rows about 4.9 m |
| `kinpo_south.csv` | South slope of Kinpo-san (32.8077 N, 130.6105 E) | Mature mikan on a hillside, in-row about 4.3 m |

All four are on the Kinpo-san mikan slopes west of Kumamoto city, about 100 km from
Fukuoka. The OpenStreetMap way ids are in each file's header comment, so you can
open the exact field in Google Earth or OSM before a site visit.

Each `*_preview.png` shows the detections drawn on the aerial image, and each
`*_plan.png` is a top-down plan. Look at the preview before trusting a layout.

Real mikan rows are tighter than the orbit reward in `orchard_env.py` can handle
(it needs roughly 5 m between trunks to fly a 3 to 5.5 m orbit). The environment
keeps the real row geometry but thins the window to `LAYOUT_MIN_TREE_DIST`.

## CSV format

```
# name=... origin_lat=... origin_lon=... source=gsi-aerial-z18 crop=mikan ...
north_m,east_m,z_m,yaw_deg,scale,crown_r_m,lat,lon
```

`north_m`/`east_m` are metres from the origin (AirSim +x is north, +y is east).
Only relative positions matter: the environment re-origins and rotates the window.

## Making a new layout

Everything runs inside `bc_env` from `bc_learning/scripts`.

From open aerial imagery (GSI ortho tiles for Japan, Esri elsewhere):

```
python make_layout.py aerial --lat 32.8711 --lon 130.6037 --radius 45 \
    --crown-r 1.8 --min-spacing 3.0 --name my_site --crop mikan
```

Lower `--veg-thresh` (for example 0.06) for small or dark young trees. Add
`--roi-kml field.kml` to only detect inside a polygon you drew in Google Earth.

From Google Earth (placemarks = trees, paths = rows, polygons = blocks to fill with rows):

```
python make_layout.py kml --file site.kml --in-row 4.0 --row 6.0 --name my_site --preview
```

Synthetic rows in the same format:

```
python make_layout.py grid --rows 2 --cols 4 --row 8 --in-row 8 --name grid_2x4
```

Inspect: `python make_layout.py show *.csv --plan`

## Imagery attribution

`gsi` tiles are the Geospatial Information Authority of Japan seamless photo
layer (国土地理院 シームレス空中写真). `esri` tiles are Esri World Imagery.
Downloaded tiles are cached in `.tile_cache/` (git-ignored).
