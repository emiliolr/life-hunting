// GEE imports
var popDens = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density"),
    povMap = ee.Image("users/el590/povmap-grdi-v1"),
    travelTimeSmall = ee.Image("users/el590/travel_time_to_cities_9"),
    protectedAreas = ee.Image("users/el590/wdpa_mask"),
    studySites = ee.FeatureCollection("users/el590/hunting_sites");

// Provided helper functions from this tutorial: https://developers.google.com/earth-engine/tutorials/community/extract-raster-values-for-points
function bufferPoints(radius, bounds) {
  return function(pt) {
    pt = ee.Feature(pt);
    return bounds ? pt.buffer(radius).bounds() : pt.buffer(radius);
  };
}

function zonalStats(ic, fc, params) {
  // Initialize internal params dictionary.
  var _params = {
    reducer: ee.Reducer.mean(),
    scale: null,
    crs: null,
    bands: null,
    bandsRename: null,
    imgProps: null,
    imgPropsRename: null
    // datetimeName: 'datetime',
    // datetimeFormat: 'YYYY-MM-dd HH:mm:ss'
  };

  // Replace initialized params with provided params.
  if (params) {
    for (var param in params) {
      _params[param] = params[param] || _params[param];
    }
  }

  // Set default parameters based on an image representative.
  var imgRep = ic.first();
  var nonSystemImgProps = ee.Feature(null)
    .copyProperties(imgRep).propertyNames();
  if (!_params.bands) _params.bands = imgRep.bandNames();
  if (!_params.bandsRename) _params.bandsRename = _params.bands;
  if (!_params.imgProps) _params.imgProps = nonSystemImgProps;
  if (!_params.imgPropsRename) _params.imgPropsRename = _params.imgProps;

  // Map the reduceRegions function over the image collection.
  var results = ic.map(function(img) {
    // Select bands (optionally rename), set a datetime & timestamp property.
    img = ee.Image(img.select(_params.bands, _params.bandsRename));
      // .set(_params.datetimeName, img.date().format(_params.datetimeFormat))
      // .set('timestamp', img.get('system:time_start'));

    // Define final image property dictionary to set in output features.
    var propsFrom = ee.List(_params.imgProps);
      // .cat(ee.List([_params.datetimeName, 'timestamp']));
    var propsTo = ee.List(_params.imgPropsRename);
      // .cat(ee.List([_params.datetimeName, 'timestamp']));
    var imgProps = img.toDictionary(propsFrom).rename(propsFrom, propsTo);

    // Subset points that intersect the given image.
    var fcSub = fc.filterBounds(img.geometry());

    // Reduce the image by regions.
    return img.reduceRegions({
      collection: fcSub,
      reducer: _params.reducer,
      scale: _params.scale,
      crs: _params.crs
    })
    // Add metadata to each feature.
    .map(function(f) {
      return f.set(imgProps);
    });
  }).flatten();

  return results;
}

// Aligning band names for all input rasters
var protectedAreas = protectedAreas.select(['first'], ['b1']);
var popDens = popDens.map(function(img){
  return img.select(['population_density'], ['b1']);
});

// Adding buffer to the points... shouldn't really be relevant given the 1km resolution
var ptsBuffer = studySites.map(bufferPoints(200, false));
print(ptsBuffer);

// Putting all datasets together into one image collection
var allRasters = ee.ImageCollection([povMap, travelTimeSmall, protectedAreas]);
var allRasters = allRasters.merge(popDens);

//  adding a dataset ID for cleanup in Python
var allRasters = allRasters.map(function(img){
  return img.set('dataset_id', img.get('system:id'));
});
print(allRasters);

// Define parameters for the zonalStats function
var params = {
  bands: ['b1'],
  imgProps: ['dataset_id'],
  reducer: ee.Reducer.mean()
};

// Extract zonal statistics per point per image
var ptsBufferStats = zonalStats(allRasters, ptsBuffer, params);
print(ptsBufferStats.first());

// Exporting to Google Drive
Export.table.toDrive({
  collection: ptsBufferStats,
  folder: 'gee_testing',
  description:'spatial_preds_study_sites',
  fileFormat: 'CSV'
});

// Mapping for testing purposes
// Map.addLayer(popDens, {min : 200, max : 1000}, 'Pop. Density');
// Map.addLayer(povMap, {min : 0, max : 100}, 'Poverty Map')
Map.addLayer(protectedAreas, {min : 0, max : 1}, 'WDPA');
Map.addLayer(ptsBuffer, {color : 'red'}, 'Reference Points');
Map.centerObject(ptsBuffer);
