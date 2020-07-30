#----------LIBRARIES--------------#

import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
import io
import streamlit.components.v1 as components
#pour plus d'info sur l'usage de components voir:
#https://github.com/Jcharis/Streamlit_DataScience_Apps/tree/master/EDA_app_with_Streamlit_Components
import os
import numpy as np
import cv2

#----------PARAMETRES--------------#

st.set_option('deprecation.showfileUploaderEncoding', False)
CONF_THRESH = 0.5
NMS_THRESH = 0.5
YOLO_CONFIG_PATH = './custom-yolov4-detector.cfg'
YOLO_WEIGHTS_PATH = './custom-yolov4-tobacco.weights'

#----------FUNCTIONS---------------#

#function to load local css file
# https://discuss.streamlit.io/t/creating-a-nicely-formatted-search-field/1804/2
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def local_html(file_name):
	with open(file_name) as f:
		return f'{f.read()}'

@st.cache(ttl=60*5,max_entries=5)
def convert_pdf_to_png(byte_object):
	png_image=convert_from_bytes(byte_object,fmt="png")
	return png_image

@st.cache(allow_output_mutation=True)
def load_the_network(yolo_cfg, yolo_weights):
	net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	layers = net.getLayerNames()
	output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	return net, output_layers

def get_predictions(net, output_layers, PIL_image):
  opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2GRAY)
  height, width = opencvImage.shape[:2]

  blob = cv2.dnn.blobFromImage(opencvImage, 0.00392, (416, 416), swapRB=True, crop=False)
  net.setInput(blob)
  layer_outputs = net.forward(output_layers)

  class_ids, confidences, b_boxes = [], [], []
  for output in layer_outputs:
      for detection in output:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]

          if confidence > CONF_THRESH:
              center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

              x = int(center_x - w / 2)
              y = int(center_y - h / 2)

              b_boxes.append([x, y, int(w), int(h)])
              confidences.append(float(confidence))
              class_ids.append(int(class_id))

  return class_ids, confidences, b_boxes, opencvImage

def NMS(confidences, b_boxes, CONF_THRESH, NMS_THRESH):
  results=cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH)
  if len(results) > 0:
  	return results.flatten().tolist()
  else:
  	return results

def draw_boxes(results, b_boxes, class_ids, opencvImage):
  classes = ['signature']
  color = (0, 0, 255)
  for box in results:
      x, y, w, h = b_boxes[box]
      cv2.rectangle(opencvImage, (x, y), (x + w, y + h), color, 2)
      cv2.putText(opencvImage, classes[class_ids[box]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
  return opencvImage

#----------APPLICATION-------------#

def main():
	"""Run this function to run the app"""

	menu = ["Home","Application"]
	st.sidebar.title("Menu")
	choice = st.sidebar.selectbox("",menu)

	if choice == "Application":

		local_css("style.css")
		st.title("Choisir un document")
		net, output_layers = load_the_network(YOLO_CONFIG_PATH,YOLO_WEIGHTS_PATH)
		file = st.file_uploader("", multiple_files=True, type=["png", "jpg", "jpeg", "pdf"])
		placeholder = st.empty()
		submit=False

		if file is not None:
			try:
				image = Image.open(file)
				placeholder.image(image,width=320)
				result_holder = st.empty()
				submit = st.button('Détecter les signatures!')
			except:
				image_png = convert_pdf_to_png(file.read())
				if len(image_png) > 1:
					pages = [str(i) + ' page' for i in range(1,len(image_png)+1)]
					selected_page = st.selectbox('Select a page', pages)
					image = image_png[pages.index(selected_page)]
					placeholder.image(image,width=320)
					result_holder = st.empty()
					submit = st.button('Détecter les signatures!')
				else:
					image = image_png[0]
					placeholder.image(image,width=320)
					result_holder = st.empty()
					submit = st.button('Détecter les signatures!')

		if submit:
			class_ids, confidences, b_boxes, opencvImage=get_predictions(net, output_layers, image)
			results=NMS(confidences, b_boxes, CONF_THRESH, NMS_THRESH)

			if len(results)==0:
				placeholder.image([image,opencvImage], width=320)
				result_holder.success("Yolo n'a détécté aucune signatures")
			else:
				opencvImage=draw_boxes(results, b_boxes, class_ids, opencvImage)
				placeholder.image([image,opencvImage], width=320)
				result_holder.success("Yolo a détécté {} signatures!".format(len(results)))

	else:
		local_css("style.css")
		st.title("Détecteur de signatures")
		st.markdown("👈 **Pour commencer choisissez _Application_ dans le menu.**")
		home_page=local_html("home.html")
		components.html(home_page, height=560)


if __name__ == "__main__":
    main()