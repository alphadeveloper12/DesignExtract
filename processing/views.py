from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import cv2 as cv
from .forms import ImageUploadForm
from .utils import process_image, process_image1, process_image_additional

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.files.get('image')

            if image_file:
                fs = FileSystemStorage()
                filename = fs.save(image_file.name, image_file)
                file_url = fs.url(filename)
                file_path = os.path.join(settings.MEDIA_ROOT, filename)

                # Read and process the image
                im = cv.imread(file_path)
                if im is None:
                    return render(request, 'upload_image.html', {'form': form, 'error': 'Could not read image'})

                # Convert to grayscale and perform Canny edge detection
                gray_img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                canny_output = cv.Canny(gray_img, 100, 200)
                canny_filename = 'canny_' + filename
                canny_file_path = os.path.join(settings.MEDIA_ROOT, canny_filename)
                cv.imwrite(canny_file_path, canny_output)
                canny_file_url = fs.url(canny_filename)

                # Process the images
                results = process_image(canny_file_path)
                results1 = process_image1(file_path)
                results_additional = process_image_additional(file_path)


                # Generate URLs for processed images
                processed_images_urls = [fs.url(image) for image in results['processed_images']]
                processed_images_url1 = [fs.url(image) for image in results1['processed_images']]
                processed_images_additional_urls = results_additional['processed_images']
                contoured_image_path = results_additional['contoured_image_path']


                return render(request, 'result.html', {
                    'image_url': file_url,
                    'processed_image_url': canny_file_url,
                    'caption': results1.get('caption', ''),
                    'precision': results1.get('precision', ''),
                    'recall': results1.get('recall', ''),
                    'f1_score': results1.get('f1_score', ''),
                    'speed': results1.get('speed', ''),
                    'processed_images': processed_images_urls,
                    'processed_images_url1': processed_images_url1,
                    'processed_images_additional': processed_images_additional_urls,
                    'contoured_image_path': contoured_image_path,
                })
    else:
        form = ImageUploadForm()
    return render(request, 'upload_image.html', {'form': form})
