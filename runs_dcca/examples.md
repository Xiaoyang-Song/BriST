
### test region 5239  (GT severity critical)
- **expert reference** [A]: Crack defect detected; region rejected (critical). CT shows two linear cracks: a 210 um diagonal (lower-left to upper-right) crack in the left side, and a 50 um diagonal (upper-left to lower-right) crack in the upper-left. Total crack length ~260 um.
- **teacher(CT)**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 240 um diagonal (lower-left to upper-right) crack in the left side, and a 100 um diagonal (lower-left to upper-right) crack in the left side. Total crack length ~340 um.
- **direct**: Critical cracking - reject. CT shows two linear cracks: a 240 um diagonal (lower-left to upper-right) crack in the centre, and a 120 um diagonal (lower-left to upper-right) crack in the centre. The two cracks intersect. Total crack length ~360 um.
- **feature_vlm**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 250 um diagonal (lower-left to upper-right) crack in the centre, and a 90 um horizontal crack in the centre. The two cracks intersect. Total crack length ~340 um.
- **hier_kd**: Crack defect detected; region rejected (significant). CT shows a 180 um vertical crack in the upper centre. Total crack length ~180 um.
- **hier_vlm**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 250 um diagonal (lower-left to upper-right) crack in the left side, and a 190 um diagonal (upper-left to lower-right) crack in the upper centre. Total crack length ~440 um.
- **selective_kd**: Crack defect detected; region rejected (critical). CT shows two linear cracks: a 260 um diagonal (lower-left to upper-right) crack in the centre, and a 190 um diagonal (lower-left to upper-right) crack in the centre. The two cracks intersect. Total crack length ~450 um.
- **token_zeroshot**: Region contains significant crack damage and should be rejected. CT shows two linear cracks: a 170 um diagonal (upper-left to lower-right) crack in the centre, and a 90 um horizontal crack in the centre. Total crack length ~260 um.

### test region 5485  (GT severity significant)
- **expert reference** [C]: Defect: crack | Count: 2 | Max length: 180 um | Orientation: vertical, diagonal (upper-left to lower-right) | Location: right side, upper-right | Intersecting: no | Severity: significant
- **teacher(CT)**: Crack defect detected; region rejected (significant). CT shows a 180 um vertical crack in the right side. Total crack length ~180 um.
- **direct**: Critical cracking - reject. CT shows two linear cracks: a 240 um diagonal (lower-left to upper-right) crack in the centre, and a 120 um diagonal (lower-left to upper-right) crack in the centre. Total crack length ~360 um.
- **feature_vlm**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 250 um diagonal (lower-left to upper-right) crack in the centre, and a 90 um horizontal crack in the centre. The two cracks intersect. Total crack length ~340 um.
- **hier_kd**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 260 um diagonal (lower-left to upper-right) crack in the right side, and a 150 um diagonal (lower-left to upper-right) crack in the centre. Total crack length ~410 um.
- **hier_vlm**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 250 um diagonal (lower-left to upper-right) crack in the right side, and a 150 um diagonal (lower-left to upper-right) crack in the upper centre. At least one crack runs to the boundary of the scanned section. Total crack length ~450 um.
- **selective_kd**: Crack defect detected; region rejected (significant). CT shows a 180 um vertical crack in the lower centre. Total crack length ~180 um.
- **token_zeroshot**: Region contains significant crack damage and should be rejected. CT shows two linear cracks: a 110 um diagonal (upper-left to lower-right) crack in the right side, and a 50 um diagonal (lower-left to upper-right) crack in the centre. Total crack length ~160 um.

### test region 5830  (GT severity critical)
- **expert reference** [C]: Defect: crack | Count: 2 | Max length: 330 um | Orientation: horizontal, horizontal | Location: lower centre, left side | Intersecting: no | Severity: critical
- **teacher(CT)**: Critical cracking - reject. CT shows two linear cracks: a 260 um horizontal crack in the centre, and a 220 um horizontal crack in the lower-left. Total crack length ~480 um.
- **direct**: Critical cracking - reject. CT shows two linear cracks: a 240 um diagonal (lower-left to upper-right) crack in the centre, and a 120 um diagonal (lower-left to upper-right) crack in the centre. Total crack length ~360 um.
- **feature_vlm**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 250 um diagonal (lower-left to upper-right) crack in the centre, and a 90 um horizontal crack in the centre. The two cracks intersect. Total crack length ~340 um.
- **hier_kd**: Crack defect detected; region rejected (critical). CT shows two linear cracks: a 260 um diagonal (lower-left to upper-right) crack in the centre, and a 220 um diagonal (lower-left to upper-right) crack in the left side. Total crack length ~480 um.
- **hier_vlm**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 160 um diagonal (upper-left to lower-right) crack in the left side, and a 130 um diagonal (upper-left to lower-right) crack in the upper centre. Total crack length ~390 um.
- **selective_kd**: Crack defect detected; region rejected (critical). CT shows two linear cracks: a 260 um diagonal (lower-left to upper-right) crack in the centre, and a 190 um diagonal (lower-left to upper-right) crack in the centre. Total crack length ~450 um.
- **token_zeroshot**: Region contains significant crack damage and should be rejected. CT shows two linear cracks: a 170 um diagonal (upper-left to lower-right) crack in the centre, and a 130 um horizontal crack in the centre. Total crack length ~300 um.

### test region 5611  (GT severity minor)
- **expert reference** [B]: Minor cracking: one crack, longest ~100 um.
- **teacher(CT)**: Crack defect detected; region rejected (minor). CT shows a 90 um diagonal (upper-left to lower-right) crack in the right side. Total crack length ~90 um.
- **direct**: Critical cracking - reject. CT shows two linear cracks: a 240 um diagonal (lower-left to upper-right) crack in the centre, and a 120 um diagonal (lower-left to upper-right) crack in the centre. Total crack length ~360 um.
- **feature_vlm**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 250 um diagonal (lower-left to upper-right) crack in the centre, and a 90 um horizontal crack in the centre. The two cracks intersect. Total crack length ~340 um.
- **hier_kd**: Region contains significant crack damage and should be rejected. CT shows a 180 um diagonal (upper-left to lower-right) crack in the lower centre. Total crack length ~180 um.
- **hier_vlm**: Region contains critical crack damage and should be rejected. CT shows two linear cracks: a 250 um diagonal (lower-left to upper-right) crack in the right side, and a 150 um diagonal (lower-left to upper-right) crack in the upper centre. At least one crack runs to the boundary of the scanned section. Total crack length ~400 um.
- **selective_kd**: Region contains minor crack damage and should be rejected. CT shows a 50 um diagonal (lower-left to upper-right) crack in the lower centre. Total crack length ~50 um.
- **token_zeroshot**: Region contains significant crack damage and should be rejected. CT shows two linear cracks: a 170 um diagonal (upper-left to lower-right) crack in the right side, and a 80 um horizontal crack in the centre. Total crack length ~250 um.

### test region 5020  (GT severity acceptable)
- **expert reference** [B]: No cracking. Two rounded pores (max ~110 um); acceptable.
- **teacher(CT)**: Crack-free; quality acceptable. CT shows two near-circular pores of ~110 and ~90 um diameter, in the left side and the upper-left; the pores overlap. Pore area fraction 10.7%.
- **direct**: No cracks detected; region acceptable. CT shows one near-circular pore of ~90 um diameter in the centre. Pore area fraction 7.9%.
- **feature_vlm**: Acceptable region without cracking. CT shows one near-circular pore of ~90 um diameter in the lower centre. Pore area fraction 4.8%.
- **hier_kd**: Crack-free; quality acceptable. CT shows two near-circular pores of ~110 and ~90 um diameter, in the lower centre and centre, partially merged into a cluster. Pore area fraction 12.7%.
- **hier_vlm**: Acceptable region without cracking. CT shows two near-circular pores of ~90 and ~70 um diameter, in the lower-left and centre, well separated. Pore area fraction 7.6%.
- **selective_kd**: Crack-free; quality acceptable. CT shows two near-circular pores of ~110 and ~90 um diameter, in the centre and upper centre, well separated. Pore area fraction 12.7%.
- **token_zeroshot**: No cracks detected; region acceptable. CT shows two near-circular pores of ~110 and ~90 um diameter, in the centre and upper-right. Pore area fraction 7.6%.

### test region 5199  (GT severity acceptable)
- **expert reference** [B]: Acceptable: two small pores, no cracks.
- **teacher(CT)**: Crack-free; quality acceptable. CT shows two near-circular pores of ~110 and ~50 um diameter, in the lower centre and centre, partially merged into a cluster. Pore area fraction 7.6%.
- **direct**: No cracks detected; region acceptable. CT shows one near-circular pore of ~90 um diameter in the centre. Pore area fraction 7.9%.
- **feature_vlm**: Acceptable region without cracking. CT shows two near-circular pores of ~110 and ~90 um diameter, in the lower-left and centre, well separated. Pore area fraction 12.7%.
- **hier_kd**: Acceptable region without cracking. CT shows two near-circular pores of ~110 and ~70 um diameter, in the centre and lower-right. Pore area fraction 9.6%.
- **hier_vlm**: Crack-free; quality acceptable. CT shows two near-circular pores of ~110 and ~70 um diameter, in the centre and right side, partially merged into a cluster. Pore area fraction 8.9%.
- **selective_kd**: No cracks detected; region acceptable. CT shows two near-circular pores of ~110 and ~70 um diameter, in the lower-right and right side, partially merged into a cluster. Pore area fraction 8.0%.
- **token_zeroshot**: No cracks detected; region acceptable. CT shows two near-circular pores of ~110 and ~90 um diameter, in the centre and lower-right. Pore area fraction 7.9%.
