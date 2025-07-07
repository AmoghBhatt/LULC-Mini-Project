// Load the Global Forest Change dataset
var gfc = ee.Image('UMD/hansen/global_forest_change_2022_v1_10');

// Updated region coordinates for Haridwar
var region = ee.Geometry.Polygon([
  [78.130, 30.020], [78.130, 29.880],
  [78.250, 29.880], [78.250, 30.020],
  [78.130, 30.020]
]);

// Sentinel-2 imagery for classification and indices
var sentinel2 = ee.ImageCollection("COPERNICUS/S2")
  .filterDate('2023-01-01', '2023-12-31')
  .filterBounds(region)
  .median();

// Calculate indices
var ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI');  // Vegetation
var ndbi = sentinel2.normalizedDifference(['B11', 'B8']).rename('NDBI'); // Urbanization
var mndwi = sentinel2.normalizedDifference(['B3', 'B11']).rename('MNDWI'); // Water

// Combine indices for classification
var inputBands = ndvi.addBands([ndbi, mndwi]);

// Training data 
var trainingData = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([78.18, 29.95]), {class: 0}),
  ee.Feature(ee.Geometry.Point([78.20, 29.97]), {class: 1}),
  ee.Feature(ee.Geometry.Point([78.19, 29.92]), {class: 2}),
  ee.Feature(ee.Geometry.Point([78.15, 29.93]), {class: 3}),
  ee.Feature(ee.Geometry.Point([78.22, 29.96]), {class: 4})
]);


// Sample regions for training 
var trainingSample = inputBands.sampleRegions({
  collection: trainingData,
  properties: ['class'],
  scale: 30
});

// Train the SVM classifier 
var classifier = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 0.5
}).train({
  features: trainingSample,
  classProperty: 'class',
  inputProperties: ['NDVI', 'NDBI', 'MNDWI']
});

// Classify the input bands
var classified = inputBands.classify(classifier);

// Validation using the same training data
var validation = trainingSample.classify(classifier);
var confusionMatrix = validation.errorMatrix('class', 'classification');

// Visualization parameters with updated colors
var classifiedVis = {
  min: 0, max: 4,
  palette: ['red', 'green', 'yellow', 'blue', 'brown']  // Non-forest, Forest, Urban, Water, Bare Land
};

// Add layers to the map
Map.centerObject(region, 12);
Map.addLayer(classified.clip(region), classifiedVis, 'Classified Map');

// Add a legend
var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px 15px'}});
legend.add(ui.Label({value: 'Legend', style: {fontSize: '18px', fontWeight: 'bold'}}));

var makeRow = function(color, name) {
  var colorBox = ui.Label('', {backgroundColor: color, padding: '8px', margin: '0 0 4px 0'});
  var description = ui.Label(name, {margin: '0 0 4px 6px'});
  return ui.Panel([colorBox, description], ui.Panel.Layout.Flow('horizontal'));
};

legend.add(makeRow('red', 'Non-forest'));
legend.add(makeRow('green', 'Forest'));
legend.add(makeRow('yellow', 'Urban'));
legend.add(makeRow('blue', 'Water'));
legend.add(makeRow('brown', 'Bare Land'));
Map.add(legend);

// Print validation metrics to the console
print('Confusion Matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa Coefficient:', confusionMatrix.kappa());

// Add pixel area to the classified image
var pixelArea = ee.Image.pixelArea();

// Initialize an object to store class names and colors
var classes = [
  {name: 'Non-forest', color: 'red', value: 0},
  {name: 'Forest', color: 'green', value: 1},
  {name: 'Urban', color: 'blue', value: 2},
  {name: 'Water', color: 'yellow', value: 3},
  {name: 'Bare Land', color: 'brown', value: 4}
];

// Function to calculate area for a specific class
function calculateClassArea(classObj) {
  // Create a mask for the class
  var classMask = classified.eq(classObj.value);
  
  // Mask the pixel area image with the class mask
  var classArea = pixelArea.updateMask(classMask).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: region,
    scale: 30,
    maxPixels: 1e9
  });

  // Convert the area to square kilometers
  var areaKm2 = ee.Number(classArea.get('area')).divide(1e6);
  
  // Print the result
  print(classObj.name + ' Area (km²):', areaKm2);
}

// Iterate over each class to calculate its area
classes.forEach(calculateClassArea);


