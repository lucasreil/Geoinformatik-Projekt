from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import rasterio
from rasterio.enums import Resampling
import rioxarray
import numpy as np

def read_raster_as_array(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)
#Load all features      
slope_path = 'slope.tif'
landcover_path = 'landcover.tif'
elevation_path = 'elevation.tif'
profile_curvature_path = 'profile_curvature.tif'
all_slides_path = 'all_slides_raster.tif'
random_noslide_path = 'random_no_slide_points.tif'
landcover_path = 'landcover_aligned.tif' 
soil_path = 'soil_aligned.tif'
litho_path = 'litho_aligned3.tif'
distance_to_roads_path = 'output_raster.tif'


# Reading each .tif file into an array
slope = read_raster_as_array(slope_path)
landcover = read_raster_as_array(landcover_path)
elevation = read_raster_as_array(elevation_path)
profile_curvature = read_raster_as_array(profile_curvature_path)
all_slides_raster = read_raster_as_array(all_slides_path)
random_noslide = read_raster_as_array(random_noslide_path)
lancover = read_raster_as_array(landcover_path)
litho = read_raster_as_array(litho_path)
soil = read_raster_as_array(soil_path)
distance_to_roads = read_raster_as_array(distance_to_roads_path)


import numpy as np

def delete_random_points(array, delete_fraction=0.9):
    # Find the indices of all non-zero points
    non_zero_indices = np.argwhere(array != 0)
    
    # Calculate the number of points to delete
    n_delete = int(len(non_zero_indices) * delete_fraction)
    
    # Randomly select indices of points to delete
    indices_to_delete = np.random.choice(range(len(non_zero_indices)), size=n_delete, replace=False)
    
    # Set the selected points to 0 (deleting them)
    for index in indices_to_delete:
        array[non_zero_indices[index][0], non_zero_indices[index][1]] = 0
        
    return array

# Assuming the random_noslide variable is already defined and contains the raster data
random_noslide_updated = delete_random_points(random_noslide)



#here you can align the rasters of the different features if necessary, they have to have the same size, projection and pixel-size

litho_xr = rioxarray.open_rasterio('litho_aligned3.tif')
slope_xr = rioxarray.open_rasterio('slope.tif')
litho_resampled = litho_xr.rio.reproject_match(slope_xr)
litho_resampled = litho_resampled.squeeze()

soil_xr = rioxarray.open_rasterio('soil_aligned.tif')

soil_resampled = soil_xr.rio.reproject_match(slope_xr) 
soil_resampled = soil_resampled.squeeze()



prec_xr = rioxarray.open_rasterio('precip_change_Ord_krig.tif')
slope_xr = rioxarray.open_rasterio('slope.tif')
prec_resampled = prec_xr.rio.reproject_match(slope_xr)
prec_resampled = litho_resampled.squeeze()





mask = all_slides_raster == 1

no_landslide_mask_adjusted = np.where(~mask, random_noslide_updated == 1, False)

combined_labels_adjusted = np.full_like(mask, -1, dtype=int)  # -1 für unlabeled
combined_labels_adjusted[mask] = 1  # Bereiche mit Erdrutschen
combined_labels_adjusted[no_landslide_mask_adjusted] = 0  # Explizit ohne Erdrutsche

is_labeled = combined_labels_adjusted != -1

# Anwenden der Filterung auf Labels
labels_filtered = combined_labels_adjusted[is_labeled]

# Anwenden der Filterung auf jedes Merkmal unter Berücksichtigung der Maskierung
masked_slope_filtered = np.where(is_labeled, slope, np.nan)[is_labeled]
masked_elevation_filtered = np.where(is_labeled, elevation, np.nan)[is_labeled]
masked_profile_curvature_filtered = np.where(is_labeled, profile_curvature, np.nan)[is_labeled]
masked_lancover_filtered = np.where(is_labeled, lancover, np.nan)[is_labeled]
masked_litho_filtered = np.where(is_labeled, litho_resampled, np.nan)[is_labeled]
masked_soil_filtered = np.where(is_labeled, soil_resampled, np.nan)[is_labeled]
masked_distance2road_filtered = np.where(is_labeled, distance_to_roads, np.nan)[is_labeled]
masked_prec_resampled_filtered = np.where(is_labeled, prec_resampled, np.nan)[is_labeled]


features_unmasked = np.stack((slope, elevation, profile_curvature, landcover, litho_resampled, soil_resampled, distance_to_roads, prec_resampled), axis=-1).reshape(-1, 8)


features = np.stack((masked_slope_filtered, masked_elevation_filtered, masked_profile_curvature_filtered, masked_lancover_filtered, masked_litho_filtered, masked_soil_filtered, masked_distance2road_filtered, masked_prec_resampled_filtered), axis=-1).reshape(-1, 8)


#features = np.stack((masked_slope, masked_elevation, masked_profile_curvature), axis=-1).reshape(-1, 3)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_filtered, test_size=0.2, random_state=42, shuffle=True)

# Insantiate and train the Raom Forest mo
model = RandomForestClassifier(n_estimators=100, random_state=42)   
model.fit(X_train, y_train)

predictions = model.predict_proba(features_unmasked)[:, 1]


from sklearn.ensemble import GradientBoostingClassifier

# Ersetze die Instanziierung des RandomForestClassifier durch GradientBoostingClassifier
model_grad_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model_grad_boost.fit(X_train, y_train)

# Vorhersagen bleiben gleich
predictions_model_grad_boost = model_grad_boost.predict_proba(features_unmasked)[:, 1]


test_probabilities = model.predict_proba(X_test)[:, 1]
test_probabilities_model_grad_boost = model_grad_boost.predict_proba(X_test)[:, 1]


import numpy as np
import matplotlib.pyplot as plt
feature_importances = model.feature_importances_
features = ['masked_slope_filtered', 'masked_elevation_filtered', 'masked_profile_curvature_filtered', 'masked_lancover_filtered', 'masked_litho_filtered', 'masked_soil_filtered', 'masked_distance2road_filtered', 'masked_prec_resampled_filtered']

friendly_feature_names = ['Slope', 'Elevation', 'Profile Curvature', 'Landcover','Lithology', 'Soil', 'Distance to Road', 'Precipitaion']
# Print the feature importances
#for feature, importance in zip(features, feature_importances):
    #print(f"{feature}: {importance}")

plt.figure(figsize=(10, 6))
plt.barh(friendly_feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Random Forest Model')
plt.savefig('feature_importance_plot_RF.png', bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
feature_importances = model_grad_boost.feature_importances_
features = ['masked_slope_filtered', 'masked_elevation_filtered', 'masked_profile_curvature_filtered', 'masked_lancover_filtered', 'masked_litho_filtered', 'masked_soil_filtered', 'masked_distance2road_filtered', 'masked_prec_resampled_filtered']
friendly_feature_names = ['Slope', 'Elevation', 'Profile Curvature', 'Landcover','Lithology', 'Soil', 'Distance to Road', 'Precipitation']

# Print the feature importances
#for feature, importance in zip(features, feature_importances):
    #print(f"{feature}: {importance}")

plt.figure(figsize=(10, 6))
plt.barh(friendly_feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Gradient Boost Model')
plt.savefig('feature_importance_plot_GB.png', bbox_inches='tight')
plt.show()

from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

# RF
log_loss_val = log_loss(y_test, test_probabilities)
brier_score_val = brier_score_loss(y_test, test_probabilities)
roc_auc_val = roc_auc_score(y_test, test_probabilities)


# GB
log_loss_val_gb = log_loss(y_test, test_probabilities_model_grad_boost)
brier_score_val_gb = brier_score_loss(y_test, test_probabilities_model_grad_boost)
roc_auc_val_gb = roc_auc_score(y_test, test_probabilities_model_grad_boost)


print("Random Forest:")
print(f"Log Loss: {log_loss_val}")
print(f"Brier Score: {brier_score_val}")
print(f"ROC AUC Score: {roc_auc_val}")

print("Model Gradient Boost:")
print(f"Log Loss: {log_loss_val_gb}")
print(f"Brier Score: {brier_score_val_gb}")
print(f"ROC AUC Score: {roc_auc_val_gb}")



from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Definieren Sie das Modell
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Definieren Sie die Metriken, die Sie evaluieren möchten
scoring = ['accuracy', 'roc_auc', 'neg_log_loss']

# Führen Sie die Kreuzvalidierung durch
cv_results = cross_validate(model, features, labels_filtered, cv=5, scoring=scoring)
 
# Ausgabe der Ergebnisse
print(f"Genauigkeit: {np.mean(cv_results['test_accuracy'])}")
print(f"ROC AUC: {np.mean(cv_results['test_roc_auc'])}")
print(f"Log Loss: {-np.mean(cv_results['test_neg_log_loss'])}")


from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Definieren Sie das Modell
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Definieren Sie die Metriken, die Sie evaluieren möchten
scoring = ['accuracy', 'roc_auc', 'neg_log_loss']

# Führen Sie die Kreuzvalidierung durch
cv_results = cross_validate(model, features, labels_filtered, cv=8, scoring=scoring)
 
# Ausgabe der Ergebnisse
print(f"Genauigkeit: {np.mean(cv_results['test_accuracy'])}")
print(f"ROC AUC: {np.mean(cv_results['test_roc_auc'])}")
print(f"Log Loss: {-np.mean(cv_results['test_neg_log_loss'])}")


######################################################## Plot Gradient Boost
predictions_reshaped_model_grad_boost = predictions_model_grad_boost.reshape(4838, 5486)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(predictions_reshaped_model_grad_boost, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Landslide Probability')
plt.title('Landslide Prediction Probabilities Gradient Boost')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

################################################################## Plot Random Forest
predictions_reshaped = predictions.reshape(4838, 5486)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(predictions_reshaped, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Landslide Probability')
plt.title('Landslide Prediction Probabilities Random Forest')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

import rasterio

# Pfad der vorhandenen Datei, von der die Metadaten kopiert werden sollen
slope_path = 'slope.tif'

# Pfad für die neue Datei, die erstellt werden soll
output_path = 'predictions_reshaped_with_prec_RF.tif'

# Öffnen der vorhandenen Datei, um die Metadaten zu lesen
with rasterio.open(slope_path) as src:
    # Kopieren der Metadaten
    meta = src.meta.copy()

# Anpassen der Metadaten für die neuen Daten
# Hinweis: Stellen Sie sicher, dass die Dimensionen von 'predictions_reshaped' mit den 'meta' Dimensionen übereinstimmen
meta['dtype'] = 'float32'  # Stellen Sie den Datentyp entsprechend Ihren Daten ein
meta['count'] = 1  # Eine Band, passen Sie dies an, falls Ihre Daten mehrere Bänder haben

# Schreiben der neuen Daten in eine TIFF-Datei mit den angepassten Metadaten
with rasterio.open(output_path, 'w', **meta) as dst:
    dst.write(predictions_reshaped, 1)

# Ausgabe des Pfades zur neuen Datei
output_path



def read_raster_metadata_and_array(file_path, chunk_size=(1024, 1024)):
    with rasterio.open(file_path) as src:
        array = src.read(1)  # Read the first band
        width = src.width
        height = src.height
    return array, width, height
    
elevation_path = 'elevation.tif'
elevation, elevation_width, elevation_height = read_raster_metadata_and_array(elevation_path)

print(f"Elevation Width: {elevation_width}, Elevation Height: {elevation_height}")