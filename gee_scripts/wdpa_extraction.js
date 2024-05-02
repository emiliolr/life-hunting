// GEE imports
var wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons"),
    studySites = ee.FeatureCollection("users/el590/hunting_sites");

// Helper function to add buffers to study sites
function bufferPoints(radius, bounds) {
  return function(pt) {
    pt = ee.Feature(pt);
    return bounds ? pt.buffer(radius).bounds() : pt.buffer(radius);
  };
}

// A function to:
//  1. Check if a study location intersects any PAs,
//  2. Check the status of the PA, and
//  3. Record the earliest year for the intersecting PAs.
function extractPAs(studyLoc){
  var singlePoint = ee.FeatureCollection([studyLoc]);

  var badStatuses = ['Proposed', 'Unknown'];
  var intersect = wdpa.filter(
    ee.Filter.and(
      ee.Filter.bounds(singlePoint),
      ee.Filter.inList('STATUS', badStatuses).not()
    )
  );

  var returnDict = {'PA_year' : null, 'PA' : intersect.size()};

  var intersectFilt = intersect.filter(ee.Filter.neq('STATUS_YR', 0));
  var earliestYear = intersectFilt.reduceColumns({
    reducer : ee.Reducer.min(),
    selectors : ['STATUS_YR']
  });
  returnDict['PA_year'] = earliestYear.get('min');

  return ee.Feature(singlePoint.first()).set(returnDict);
}

// Adding a buffer to study sites and extracting PAs
var studySitesBuffer = studySites.map(bufferPoints(200, false));
var studySitesPAs = studySitesBuffer.map(extractPAs);

Export.table.toDrive({
  collection: studySitesPAs,
  folder: 'gee_testing',
  description:'wdpa_study_sites',
  fileFormat: 'CSV'
});
