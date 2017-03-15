Hi everyone,

GazeEstimator as can be understood from its name, is a gaze estimation project which works on linux and it uses Dlib and opencv libraries.

In the project, user's face is captured from a web cam and the application makes callibration by wanting to look some specific poinits in the screen from the user. The user's pupil's center coordinate according to eye corner is used as feature set and the feature set is trained in a ann(artificial neural network). Then, by using the ann and feature sets that captured when the user look at the calibration points, the app tries to estimate where the user looks at.  

There is a demo video in this link.
https://youtu.be/mbCpORGjFhw
