from spectralmatch import (
    compare_spatial_spectral_difference_individual_bands,
    compare_image_spectral_profiles_pairs,
    compare_image_spectral_profiles,
    compare_spatial_spectral_difference_average)

compare_spatial_spectral_difference_individual_bands(
    (
    '/image/a.tif',
    '/image/b.tif'),
    '/output.png'
)


compare_image_spectral_profiles_pairs(
    {
        'Image A': [
            '/image/before/a.tif',
            'image/after/a.tif'
        ],
        'Image B': [
            '/image/before/b.tif',
            '/image/after/b.tif'
        ]
    },
    '/output.png'
)


compare_image_spectral_profiles(
    {
        'Image A': 'image/a.tif',
        'Image B': '/image/b.tif'
    },
    "/output.png",
    "Digital Number Spectral Profile Comparison",
    'Band',
    'Digital Number(0-2,047)',

)


compare_spatial_spectral_difference_average(
    [
        '/image/a.tif',
        '/image/a.tif'
     ],
    '/output.png'
)