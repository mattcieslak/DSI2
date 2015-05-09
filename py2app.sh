#!/bin/bash

# add TVTK classes zip file to library; fixes: 'ImportError: TVTK not built properly. Unable to find either a directory: /home/jvb/git/infobiotics-dashboard/dist/library.zip/tvtk/tvtk_classes or a file: /home/jvb/git/infobiotics-dashboard/dist/library.zip/tvtk/tvtk_classes.zip with the TVTK classes.'
echo "running py2app"
python setupApp.py py2app -O1
if [ ! -d dist/DSI2.app ]; then
echo ERROR no app built
exit 1
fi
echo "unzipping tvtk_classes.zip in site-packages.zip"
unzip -q dist/DSI2.app/Contents/Resources/lib/python2.7/site-packages.zip -d dist/DSI2.app/Contents/Resources/lib/python2.7/site-packages
rm dist/DSI2.app/Contents/Resources/lib/python2.7/site-packages.zip
unzip -q dist/DSI2.app/Contents/Resources/lib/python2.7/site-packages/tvtk/tvtk_classes.zip -d dist/DSI2.app/Contents/Resources/lib/python2.7/site-packages/tvtk/
rm dist/DSI2.app/Contents/Resources/lib/python2.7/site-packages/tvtk/tvtk_classes.zip

echo Copying data files to app bundle
mkdir -p dist/DSI2.app/Contents/Resources/dsi2
cp -r dsi2/example_data dist/DSI2.app/Contents/Resources/dsi2/
cp -r /usr/local/lib/python2.7/site-packages/sklearn dist/DSI2.app/Contents/Resources/lib/python2.7/site-packages

cd dist
tar cvfz DSI2.app.tar.gz DSI2.app
