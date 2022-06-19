import numpy as np


def standardize_bbox(pcl):
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    # print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]

    # Move down onto the floor
    
    m = result.min(0)
    m[:2] = 0
    m[2] = m[2] + 0.3
    result = result - m
    
    return result


def downsample(pcl, colors, n_points):
    pt_indices = np.random.choice(pcl.shape[0], n_points, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    colors = colors[pt_indices]
    return pcl, colors


def select_downsample(pcl, noise, n_points, low_noise=False):
    if low_noise:
        idx = np.argpartition(noise.flatten(), n_points)
        pcl = pcl[idx[:n_points]] # n by 3
    else:
        np.random.shuffle(pcl)
        pcl = pcl[:n_points]
    return pcl



def xml_head():
    return """
<scene version="0.5.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="2.5,0,0.9" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="ldrfilm">
            <integer name="width" value="480"/>
            <integer name="height" value="480"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""


def xml_tail():
    return """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

# <emitter type="constant">
#     <rgb value="1.000000 1.000000 1.000000" name="radiance"/>
# </emitter>

def xml_sphere(x, y, z, r, g, b, radius=0.0075):
    tmpl = """
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
    return tmpl.format(radius, x, y, z, r, g, b)


# def xml_icosphere(x, y, z, r, g, b, radius=0.0075):
#     tmpl = """
#     <shape type="ply">
#         <string name="filename" value="icosphere.ply"/>
#         <transform name="toWorld">
#             <translate x="{}" y="{}" z="{}"/>
#         </transform>
#         <bsdf type="diffuse">
#             <rgb name="reflectance" value="{},{},{}"/>
#         </bsdf>
#     </shape>
# """
#     return tmpl.format(x, y, z, radius/2, r, g, b)


def make_xml(pcl, color, radius, flipX=False, flipY=False, flipZ=False, max_points=None):
    xml = ''
    xml += xml_head()

    pcl = standardize_bbox(pcl)
    print(pcl.shape)
    if max_points is not None:
        pcl, color = downsample(pcl, color, max_points)
    for p, c in zip(pcl, color):
        x, y, z = p
        r, g, b = c
        if flipX: x = -x
        if flipY: y = -y
        if flipZ: z = -z
        xml += xml_sphere(x, y, z, r, g, b, radius)
    
    xml += xml_tail()

    return xml
