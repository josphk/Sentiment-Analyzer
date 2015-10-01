# LED Emotion

LED Emotion is a project that enables a Raspberry Pi to be led by emotions, but more importantly to understand and respond to different facial expressions.

It integrates OpenCV and Haar Cascade classifiers to recognize the user's facial region in order to extract feature vectors based on pixel intensity. Once extracted, it is processed against a trained linear SVM classifier. The Raspberry Pi responds with a blinking green or red LED, depending on whether the user was smiling or frowning.

Check out the video [here!](http://y2u.be/YwZT6tT7NHk)
