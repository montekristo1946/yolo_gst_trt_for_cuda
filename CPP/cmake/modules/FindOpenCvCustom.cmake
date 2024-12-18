include(cmake/modules/find_library_create_target.cmake)

macro(FindOpenCvCustom OpenCV_LIBS OpenCV_LIB_DIR)
    message(STATUS "--FindOpenCvCustom RUNNING --")
    find_library_create_target(opencv_calib3d "opencv_calib3d" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_core "opencv_core" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_dnn "opencv_dnn" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_features2d "opencv_features2d" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_flann "opencv_flann" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_gapi "opencv_gapi" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_highgui "opencv_highgui" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_imgcodecs "opencv_imgcodecs" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_imgproc "opencv_imgproc" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_ml "opencv_ml" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_objdetect "opencv_objdetect" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_photo "opencv_photo" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_stitching "opencv_stitching" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_video "opencv_video" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_videoio "opencv_videoio" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_alphamat "opencv_alphamat" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_aruco "opencv_aruco" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_bgsegm "opencv_bgsegm" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_bioinspired "opencv_bioinspired" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_ccalib "opencv_ccalib" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudaarithm "opencv_cudaarithm" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudabgsegm "opencv_cudabgsegm" SHARED ${OpenCV_LIB_DIR})
#    find_library_create_target(opencv_cudacodec "opencv_cudacodec" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudafeatures2d "opencv_cudafeatures2d" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudafilters "opencv_cudafilters" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudaimgproc "opencv_cudaimgproc" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudalegacy "opencv_cudalegacy" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudaobjdetect "opencv_cudaobjdetect" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudaoptflow "opencv_cudaoptflow" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudastereo "opencv_cudastereo" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudawarping opencv_cudawarping SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_cudev "opencv_cudev" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_datasets "opencv_datasets" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_dnn_objdetect "opencv_dnn_objdetect" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_dnn_superres "opencv_dnn_superres" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_dpm "opencv_dpm" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_face "opencv_face" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_freetype "opencv_freetype" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_fuzzy "opencv_fuzzy" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_hdf "opencv_hdf" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_hfs "opencv_hfs" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_img_hash "opencv_img_hash" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_intensity_transform "opencv_intensity_transform" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_line_descriptor "opencv_line_descriptor" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_mcc "opencv_mcc" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_optflow "opencv_optflow" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_phase_unwrapping "opencv_phase_unwrapping" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_plot "opencv_plot" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_quality "opencv_quality" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_rapid "opencv_rapid" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_reg "opencv_reg" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_rgbd "opencv_rgbd" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_saliency "opencv_saliency" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_sfm "opencv_sfm" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_shape "opencv_shape" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_signal "opencv_signal" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_stereo "opencv_stereo" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_structured_light "opencv_structured_light" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_superres "opencv_superres" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_surface_matching "opencv_surface_matching" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_text "opencv_text" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_tracking "opencv_tracking" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_videostab "opencv_videostab" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_wechat_qrcode "opencv_wechat_qrcode" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_xfeatures2d "opencv_xfeatures2d" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_ximgproc "opencv_ximgproc" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_xobjdetect "opencv_xobjdetect" SHARED ${OpenCV_LIB_DIR})
    find_library_create_target(opencv_xphoto "opencv_xphoto" SHARED ${OpenCV_LIB_DIR})

    set(${OpenCV_LIBS} opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann opencv_gapi
            opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect opencv_photo opencv_stitching
            opencv_video opencv_videoio opencv_alphamat opencv_aruco opencv_bgsegm opencv_bioinspired opencv_ccalib
            opencv_cudaarithm opencv_cudabgsegm
#            opencv_cudacodec
            opencv_cudafeatures2d opencv_cudafilters
            opencv_cudaimgproc opencv_cudalegacy opencv_cudaobjdetect opencv_cudaoptflow opencv_cudastereo
            opencv_cudawarping opencv_cudev opencv_datasets opencv_dnn_objdetect opencv_dnn_superres
            opencv_dpm opencv_face opencv_freetype opencv_fuzzy opencv_hdf opencv_hfs opencv_img_hash
            opencv_intensity_transform opencv_line_descriptor opencv_mcc opencv_optflow opencv_phase_unwrapping
            opencv_plot opencv_quality opencv_rapid opencv_reg opencv_rgbd opencv_saliency opencv_sfm
            opencv_shape opencv_signal opencv_stereo opencv_structured_light opencv_superres opencv_surface_matching
            opencv_text opencv_tracking opencv_videostab opencv_wechat_qrcode opencv_xfeatures2d opencv_ximgproc
            opencv_xobjdetect opencv_xphoto)
endmacro()