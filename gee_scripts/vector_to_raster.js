// GEE imports
var currentWDPA = ee.FeatureCollection("WCMC/WDPA/current/polygons"),
    oldestWDPA = ee.FeatureCollection("WCMC/WDPA/201707/polygons");

// Adding a protected field to hook onto for rasterization
var currentWDPA = currentWDPA.map(function(feat){
   return feat.set('protected', 1);
});
var oldestWDPA = oldestWDPA.map(function(feat){
   return feat.set('protected', 1);
});

// Getting a global binary raster for current and historical PAs
var wdpaMaskCurrent = currentWDPA.reduceToImage(['protected'], ee.Reducer.first())
                                 .unmask();
var wdpaMaskOldest = oldestWDPA.reduceToImage(['protected'], ee.Reducer.first())
                               .unmask();

// Checking global disagreement between current and historical masks
var currentVSOldest = wdpaMaskCurrent.and(wdpaMaskOldest.not());
print(currentVSOldest);

// Exporting as a GEE asset
var projection = wdpaMaskOldest.projection().getInfo();

Export.image.toAsset({
    image: wdpaMaskOldest,
    description: 'wdpa_mask_2017',
    crs: projection.crs,
    scale: 1000,
    maxPixels: 1e9
  });

// Visualizing on the GEE map
Map.addLayer(wdpaMaskCurrent, {min : 0, max : 1}, 'wdpaMask');
Map.addLayer(currentVSOldest, {min : 0, max : 1}, 'wdpaDiff');
// Map.addLayer(oldestWDPA, {color: 'red'}, 'wdpa');
