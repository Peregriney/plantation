#Python GEE functions to export CHM, S-2, S-1 from Google Earth Engine
import ee
import time
ee.Authenticate(force=True)
ee.Initialize()

africa =  ee.Geometry.Polygon(
        [[[-24.525987836328152, 43.79652618083095],
          [-24.525987836328152, -42.072078680554775],
          [52.46619966367182, -42.072078680554775],
          [52.46619966367182, 43.79652618083095]]])

table = ee.FeatureCollection("projects/ee-peregriney/assets/Descals2020_GlobalOilPalmLayer_2019")
forests_oil = table.filterBounds(geometry).filter(ee.Filter.eq('Class',3))
table = table.filter(ee.Filter.lt('Class',3))
small_oil = table.filterBounds(geometry).filter(ee.Filter.eq('Class',2))
industrial_oil = table.filterBounds(geometry).filter(ee.Filter.eq('Class',1))

orchlen = table.size().getInfo()
table = table.toList(orchlen)

forlen = forests_oil.size().getInfo()
forests_oil = forests_oil.toList(forlen)


##CHM Processing
chm = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
def export_images_CHM(feature, pathplt, pathfor):
  """
  Export each CHM image.
  """
    try: 
        geometry = feature.geometry()
        centroid = geometry.centroid()
        lon = centroid.getInfo()['coordinates'][0]
        lat = centroid.getInfo()['coordinates'][1]
        coordinates_string = f"Lat_{lat}_Lon_{lon}"
        bounding_box = ee.Geometry.Point(lon,lat).buffer(box_size/2).bounds()
        image = chm.filterBounds(bounding_box).mosaic()
        clipped_image = image.clip(bounding_box)
        
        
        bucket = 'orchard-forest-l2' 
        if plantations:
            folder = pathplt  
        else:
            folder = pathfor

      export_params = {
          'image': clipped_image,
          'description': 'Image_' + coordinates_string,
          'bucket': bucket,
          'fileNamePrefix': folder + '/Image_' + coordinates_string,
          'scale': 1, 
          'fileFormat': 'GeoTIFF',
        }

        task = ee.batch.Export.image.toCloudStorage(**export_params)
        task.start()
        return clipped_image
    except:
        raise


## Sentinel-2 Processing
BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
orchards = ee.FeatureCollection('projects/ee-peregriney/assets/Descals2023_GlobalCoconutLayer_2020')#
orchards = orchards.filter(ee.Filter.Or(ee.Filter.eq('Class', 5), ee.Filter.Or(ee.Filter.eq('Class', 4),ee.Filter.Or(ee.Filter.eq('Class', 2),ee.Filter.eq('Class', 2)))))
orch_with_random = orchards.randomColumn('random')

orchard_pts = orch_with_random.filter(ee.Filter.lt('random', 0.5)).toList(orchards.size().getInfo())

forests = ee.FeatureCollection('projects/ee-peregriney/assets/Descals2023_GlobalCoconutLayer_2020').filterBounds(africa);
forests = forests.filter(ee.Filter.eq('Class', 1))
for_with_random = forests.randomColumn('random')

forest_pts = for_with_random.filter(ee.Filter.lt('random', 0.5)).toList(forests.size().getInfo())

s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select(BANDS)

start_date = '2023-01-01'
end_date = '2023-01-31'

start_date_jul = '2023-07-01'
end_date_jul = '2023-07-31'
box_size = 240
export_tasks = []

def export_images(feature):
  """
  Function to GEE export each image feature from S2 (40x40 pixel box around centroid
  """
    geometry = feature.geometry()
    try:

      lon = feature.get('lon').getInfo()#
      lat = feature.get('lat').getInfo()#
      coordinates_string = f"Lat_{lat}_Lon_{lon}"
      bounding_box = ee.Geometry.Point(lon,lat).buffer(box_size/2).bounds()

      filtered_s2 = s2.filterDate(start_date, end_date).filterBounds(bounding_box)
      image = filtered_s2.sort('CLOUDY_PIXEL_PERCENTAGE').first()
      clipped_image = image.clip(bounding_box)
      bucket = 'orchard-forest-l2' 
      if plantations:
        folder = 'oilpalmorchards2-50m-JAN'  
      else:
        folder = 'oilpalmforests2-50m-JAN'

      export_params = {
          'image': clipped_image,
          'description': 'Image_' + coordinates_string,
          'bucket': bucket,
          'fileNamePrefix': folder + '/Image_' + coordinates_string,
          'scale': 1, 
          'fileFormat': 'GeoTIFF',
      }

      task = ee.batch.Export.image.toCloudStorage(**export_params)
      task.start()
        
      filtered_s2 = s2.filterDate(start_date_jul, end_date_jul).filterBounds(bounding_box)
      image = filtered_s2.sort('CLOUDY_PIXEL_PERCENTAGE').first()
      clipped_image = image.clip(bounding_box)
      export_params = {
          'image': clipped_image,
          'description': 'Image_Jul_' + coordinates_string,
          'bucket': bucket,
          'fileNamePrefix': folder + '/Image_Jul_' + coordinates_string,
          'scale': 1,            'fileFormat': 'GeoTIFF',
      }

      # Start the export
      task = ee.batch.Export.image.toCloudStorage(**export_params)
      task.start()

      return clipped_image

    except:
      raise


## Sentinel-1 Processing
def export_images_csv(featureset, folderpathorchard, folderpathforest,startnum):
  """
  Code to process tabular Sentinel-1 features extracted from patch centroid
  """

    global submitted_tasks
    task_limit = 3000  
    # Filter Sentinel-1 data (VV and VH polarizations)
    sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterDate(start_dates[0], end_dates[0]) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

    
    # Separate ascending and descending orbit directions
    vhAscending = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    vhDescending = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    
    # Combine ascending and descending for VH and VV polarizations and calculate mean
    sentinel1_VH = ee.ImageCollection(vhAscending.select('VH').merge(vhDescending.select('VH'))).mean()
    sentinel1_VV = ee.ImageCollection(vhAscending.select('VV').merge(vhDescending.select('VV'))).mean()
    
    # Calculate entropy for the VH band using a 4-pixel radius kernel, stdev for VH band
    square_kernel = ee.Kernel.square(radius=4)
    sentinel1_VH_entropy = sentinel1_VH.int().entropy(square_kernel)
    sentinel1_VH_stdDev = ee.ImageCollection(vhAscending.select('VH').merge(vhDescending.select('VH'))).reduce(ee.Reducer.stdDev())
    
    # Extract VV, VH, entropy, and standard deviation values for the orchards
    def extract_features(feature):
        point = feature.geometry()
        lon = feature.geometry().getInfo()['coordinates'][0]
        lat = feature.geometry().getInfo()['coordinates'][1]
        
        # Reduce regions to get VV, VH, entropy, and stdDev values
        vv_value = sentinel1_VV.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).get('VV')
        vh_value = sentinel1_VH.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).get('VH')
        vh_entropy_value = sentinel1_VH_entropy.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).get('VH')
        vh_stdDev_value = sentinel1_VH_stdDev.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).get('VH_stdDev')
        
        # Add VV, VH, entropy, stdDev, and the point's coordinates to the feature properties
        return feature.set({
            'VV': vv_value,
            'VH': vh_value,
            'VH_entropy': vh_entropy_value,
            'VH_stdDev': vh_stdDev_value,
            'latitude': lat,
            'longitude': lon
        })
    
        # Get the total number of points in the feature set
    orchlen = featureset.size().getInfo()
    batch_size = 1000  # Set the batch size to 1000
    print('Total number of points in this collection:', orchlen)
    
    # Function to export a batch of features
    def export_batch(batch_num, batch_features):
        # Create a FeatureCollection for the current batch
        extracted = ee.FeatureCollection(batch_features)
        
        # Define the Google Cloud Storage bucket path
        bucket = 'orchard-forest-l2'  # Replace with your bucket name
        
        # Determine the folder based on plantations
        if plantations:
            folder = f"{folderpathorchard}-batch_{batch_num}"  # Different folder for each batch
        else:
            folder = f"{folderpathforest}-batch_{batch_num}"
        
        # Export the batch to Google Cloud Storage as a CSV
        task = ee.batch.Export.table.toCloudStorage(
            collection=extracted,
            bucket=bucket,
            fileNamePrefix=folder,
            fileFormat='CSV',
        )
        
        # Start the export task
        task.start()
        print(f"Started export task for batch {batch_num}")
    
    # Loop through the features and divide them into batches
    for start in range(startnum, orchlen, batch_size):

        end = min(start + batch_size, orchlen)
        print(f'Processing batch: {start} to {end}')
        
        batch_features = []
        for i in range(start, end):
            if i % 20 == 0:
                print(f"Processing feature {i}")
            ft = ee.Feature(featureset.get(i))
            batch_features.append(extract_features(ft))
        
        # Export the current batch
        export_batch(start // batch_size, batch_features)


#Example CHM Export Task
box_size = 240
export_tasks = []
submitted_tasks = 0

print('oil forests')
plantations = False
for i in range(forlen):
  if i % 250 == 0:
      print(i)
  ft = ee.Feature(forests_oil.get(i))
  export_images_CHM(ft, 'chmorchard-224m/Descalsoil','chmforest-224m/Descalsoil')
print('finished oil forests')


