extern crate vez_rs;
use vez_rs::*;

extern crate glfw;

use glfw::{Action, Context, Key};

extern crate ultraviolet;
use ultraviolet::projection::rh_ydown::perspective_vk;
use ultraviolet::*;

use std::fs;

extern crate wavefront_obj;
use wavefront_obj::obj::{self, *};

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));

    let (mut width, mut height) = (1280, 720);

    let (mut window, events) = glfw
        .create_window(width, height, "vez-sys", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.set_key_polling(true);

    assert!(glfw.vulkan_supported());
    let required_extensions = glfw.get_required_instance_extensions().unwrap_or(vec![]);

    let validation_layers = vec!["VK_LAYER_LUNARG_standard_validation".to_string()];

    let instance = InstanceBuilder::new()
        .with_layers(&validation_layers)
        .with_extensions(&required_extensions)
        .build()
        .expect("Failed to create instance");

    let physical_devices = instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices");

    let logical_device = physical_devices[0]
        .logical_device_builder()
        .with_extensions(&["VK_KHR_swapchain".to_string()])
        .build()
        .expect("Failed to create logical device");

    let mut frame_data = FrameData::new(&instance, &logical_device, &window);

    let graphics_queue = logical_device.get_graphics_queue(0);

    let vertex_shader_source =
       "#version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(binding = 0) uniform UniformStruct {
            mat4 model_matrix;
            mat4 view_matrix;
            mat4 proj_matrix;
        } ubo;

        layout(location = 0) in vec3 inPosition;
        layout(location = 1) in vec3 inNormal;
        layout(location = 2) in vec3 inColor;

        layout(location = 0) out vec3 fragColor;
        layout(location = 1) out vec3 outWorldPosition;
        layout(location = 2) out vec3 outNormal;

        void main() {
            outWorldPosition = (ubo.model_matrix * vec4(inPosition, 1.0)).xyz;
            gl_Position = (ubo.proj_matrix * ubo.view_matrix * ubo.model_matrix * vec4(inPosition, 1.0));
            fragColor = inColor;
            outNormal = (ubo.model_matrix * vec4(inNormal, 1.0)).xyz;
        }";

    let frag_shader_source = "#version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(location = 0) in vec3 inColor;
        layout(location = 1) in vec3 inWorldPosition;
        layout(location = 2) in vec3 inNormal;
        
        layout(location = 0) out vec4 outColor;

        void main() {

            vec3 n = normalize(inNormal);
            vec3 l_dir = vec3(0,-1,0);
            float intensity = max(dot(n,l_dir), 0.0);

            vec3 ambient = vec3(0.2, 0.2, 0.2);

            // outColor = vec4(max(intensity * inColor, ambient), 1.0);
            // outColor = vec4(inColor, 1.0);
            outColor = vec4(inNormal, 1.0);
       }";

    let vertex_shader_module = logical_device
        .create_shader_module(
            ShaderStage::VERTEX,
            ShaderCode::GLSL(vertex_shader_source.to_string()),
            "main",
        )
        .expect("failed to create vertex shader");

    let fragment_shader_module = logical_device
        .create_shader_module(
            ShaderStage::FRAGMENT,
            ShaderCode::GLSL(frag_shader_source.to_string()),
            "main",
        )
        .expect("failed to create fragment shader");

    let graphics_pipeline = logical_device
        .create_graphics_pipeline(&[&vertex_shader_module, &fragment_shader_module])
        .expect("failed to create graphics pipeline");

    let obj_file =
        fs::read_to_string("examples/assets/monkey.obj").expect("failed to open monkey.obj");

    let obj = obj::parse(obj_file).expect("failed to parse obj");

    let first_object = &obj.objects[0];

    #[repr(C)]
    struct Vertex {
        position: [f32; 3],
        normal: [f32; 3],
        color: [f32; 3],
    }

    let positions_iter = first_object.vertices.iter();
    let normals_iter = first_object.normals.iter();

    let vertices: Vec<Vertex> = positions_iter
        .zip(normals_iter)
        .map(|(pos, norm)| Vertex {
            position: [pos.x as _, pos.y as _, pos.z as _],
            normal: [norm.x as _, norm.y as _, norm.z as _],
            color: [1.0, 1.0, 0.0],
        })
        .collect();

    let vertex_buffer = logical_device
        .create_buffer(
            (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
            MemoryUsage::CPU_TO_GPU,
            BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
        )
        .unwrap();

    logical_device.buffer_sub_data(&vertex_buffer, &vertices, 0);

    let triangles: Vec<[usize; 3]> = first_object
        .geometry
        .iter()
        .map(|g| {
            g.shapes.iter().filter_map(|s| match s.primitive {
                Primitive::Triangle(a, b, c) => Some([a.0, b.0, c.0]),
                _ => None,
            })
        })
        .flatten()
        .collect();

    let indices: Vec<u32> = triangles
        .iter()
        .flat_map(|tri| tri.iter())
        .copied()
        .map(|i| i as u32)
        .collect();

    let index_buffer = logical_device
        .create_buffer(
            (std::mem::size_of::<u32>() * indices.len()) as u64,
            MemoryUsage::CPU_TO_GPU,
            BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
        )
        .unwrap();

    logical_device.buffer_sub_data(&index_buffer, &indices, 0);

    #[derive(Debug, Copy, Clone)]
    #[repr(C, align(16))]
    struct UniformData {
        model: Mat4,
        view: Mat4,
        proj: Mat4,
    }

    let mut eye = Vec3::new(0., -1., 5.);
    let mut at = Vec3::new(0., 0., -1.);
    let mut up = Vec3::new(0., 1., 0.);

    let mut uniform_data = UniformData {
        model: Mat4::from_euler_angles(3.14, 0.3, 0.0),
        view: Mat4::look_at(eye, at, up),
        proj: perspective_vk(45.0, width as f32 / height as f32, 0.01, 10000.0),
    };

    let uniform_buffer = logical_device
        .create_buffer(
            std::mem::size_of::<UniformData>() as u64,
            MemoryUsage::CPU_TO_GPU,
            BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
        )
        .unwrap();

    logical_device.buffer_sub_data(&uniform_buffer, &[uniform_data.clone()], 0);

    let command_buffers = logical_device
        .allocate_command_buffers(&graphics_queue, 1)
        .unwrap();
    let command_buffer = &command_buffers[0];

    //Store as a closure so we can call on Resize
    let record_command_buffer = |frame_data: &FrameData, width: u32, height: u32| {
        command_buffer.begin_recording(CommandBufferUsage::SIMULTANEOUS_USE, |command_recorder| {
            //TODO: Viewport struct
            command_recorder.set_scissor(0, 0, width, height);
            command_recorder.set_viewport(
                0,
                Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as _,
                    height: height as _,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
            );

            //TODO: attachment info, loadOp, storeOp enums
            command_recorder.begin_render_pass(
                &frame_data.framebuffer,
                &[
                    VezAttachmentInfo {
                        loadOp: VkAttachmentLoadOp::VK_ATTACHMENT_LOAD_OP_CLEAR,
                        storeOp: VkAttachmentStoreOp::VK_ATTACHMENT_STORE_OP_STORE,
                        clearValue: ClearValue::Color(ClearColor::Float([0.39, 0.58, 0.93, 1.0]))
                            .into(), //TODO: make more ergonomic
                    },
                    VezAttachmentInfo {
                        loadOp: VkAttachmentLoadOp::VK_ATTACHMENT_LOAD_OP_CLEAR,
                        storeOp: VkAttachmentStoreOp::VK_ATTACHMENT_STORE_OP_DONT_CARE,
                        clearValue: ClearValue::DepthStencil(ClearDepthStencil {
                            depth: 1.0,
                            stencil: 0,
                        })
                        .into(), //TODO: make more ergonomic
                    },
                ],
                || {
                    //TODO: rustified struct
                    command_recorder.set_depth_stencil_state(&VezDepthStencilState {
                        pNext: std::ptr::null(),
                        depthTestEnable: true as _,
                        depthWriteEnable: true as _,
                        depthCompareOp: VkCompareOp::VK_COMPARE_OP_LESS_OR_EQUAL,
                        depthBoundsTestEnable: false as _,
                        stencilTestEnable: false as _,
                        front: VezStencilOpState {
                            failOp: VkStencilOp::VK_STENCIL_OP_KEEP,
                            passOp: VkStencilOp::VK_STENCIL_OP_KEEP,
                            depthFailOp: VkStencilOp::VK_STENCIL_OP_KEEP,
                            compareOp: VkCompareOp::VK_COMPARE_OP_NEVER,
                        },
                        back: VezStencilOpState {
                            failOp: VkStencilOp::VK_STENCIL_OP_KEEP,
                            passOp: VkStencilOp::VK_STENCIL_OP_KEEP,
                            depthFailOp: VkStencilOp::VK_STENCIL_OP_KEEP,
                            compareOp: VkCompareOp::VK_COMPARE_OP_NEVER,
                        },
                    });

                    command_recorder.bind_pipeline(&graphics_pipeline);
                    command_recorder.bind_vertex_buffer(0, &vertex_buffer, 0);
                    command_recorder.bind_index_buffer(&index_buffer, 0, IndexType::U32);
                    command_recorder.bind_buffer(&uniform_buffer, 0, VK_WHOLE_SIZE as _, 0, 0, 0);
                    command_recorder.draw_indexed(0..indices.len() as _, 0..1, 0);
                },
            ); //RenderPass automatically ends here
        }); //Command buffer automatically ends recording here
    };
    record_command_buffer(&frame_data, width, height);

    loop {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true)
                }
                _ => {}
            }
        }

        if window.should_close() {
            break;
        }

        let (new_width, new_height) = {
            let (new_width, new_height) = window.get_size();
            (new_width as u32, new_height as u32)
        };

        if new_width == 0 || new_height == 0 {
            continue;
        }

        if new_width != width || new_height != height {
            logical_device.wait_idle();
            width = new_width;
            height = new_height;
            let old_frame_data = frame_data;
            old_frame_data.destroy(&instance, &logical_device);
            frame_data = FrameData::new(&instance, &logical_device, &window);

            record_command_buffer(&frame_data, width, height);
        }

        uniform_data.model = uniform_data.model * Mat4::from_euler_angles(0.005, 0.001, 0.01);

        logical_device.buffer_sub_data(&uniform_buffer, &[uniform_data.clone()], 0);

        let fence = graphics_queue.submit(&[QueueSubmission {
            command_buffers: &[&command_buffer],
        }]);
        graphics_queue.present(&[(&frame_data.swapchain, &frame_data.image)]);
    }

    logical_device.wait_idle();
    logical_device.free_command_buffers(&command_buffers);
    logical_device.destroy_buffer(vertex_buffer);
    logical_device.destroy_buffer(index_buffer);
    logical_device.destroy_buffer(uniform_buffer);
    logical_device.destroy_pipeline(graphics_pipeline);
    logical_device.destroy_shader_module(vertex_shader_module);
    logical_device.destroy_shader_module(fragment_shader_module);

    frame_data.destroy(&instance, &logical_device);

    logical_device.destroy();
    instance.destroy();
}

struct FrameData {
    surface: VkSurfaceKHR,
    swapchain: Swapchain,
    image: Image,
    image_view: ImageView,
    depth_image: Image,
    depth_image_view: ImageView,
    framebuffer: Framebuffer,
}

impl FrameData {
    fn new(
        instance: &Instance,
        logical_device: &LogicalDevice,
        window: &glfw::Window,
    ) -> FrameData {
        let surface = unsafe {
            let mut surface = VK_NULL_HANDLE as u64;
            let result = glfw::ffi::glfwCreateWindowSurface(
                instance.raw as usize,
                window.window_ptr(),
                std::ptr::null(),
                &mut surface,
            );

            surface as VkSurfaceKHR
        };

        let swapchain = logical_device
            .create_swapchain(
                &surface,
                VkFormat::VK_FORMAT_B8G8R8A8_UNORM,
                VkColorSpaceKHR::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                true, //Triple Buffer
            )
            .expect("Failed to Create Swapchain");

        swapchain.set_vsync_enabled(true);

        let format = swapchain.get_surface_format().format;

        let (width, height) = {
            let (width, height) = window.get_size();
            (width as u32, height as u32)
        };

        let image = logical_device
            .create_image(
                width,
                height,
                format,
                MemoryUsage::GPU_ONLY,
                ImageUsage::TRANSFER_SRC | ImageUsage::COLOR_ATTACHMENT,
            )
            .unwrap();

        let image_view = logical_device
            .create_image_view(&image, VkImageViewType::VK_IMAGE_VIEW_TYPE_2D, format)
            .unwrap();

        let depth_format = VkFormat::VK_FORMAT_D32_SFLOAT;

        let depth_image = logical_device
            .create_image(
                width,
                height,
                depth_format,
                MemoryUsage::GPU_ONLY,
                ImageUsage::TRANSFER_SRC | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            )
            .unwrap();

        let depth_image_view = logical_device
            .create_image_view(
                &depth_image,
                VkImageViewType::VK_IMAGE_VIEW_TYPE_2D,
                depth_format,
            )
            .unwrap();

        let framebuffer = logical_device
            .create_framebuffer(&[&image_view, &depth_image_view], width, height, 1)
            .unwrap();

        FrameData {
            surface,
            swapchain,
            image,
            image_view,
            depth_image,
            depth_image_view,
            framebuffer,
        }
    }

    fn destroy(self, instance: &Instance, logical_device: &LogicalDevice) {
        logical_device.destroy_framebuffer(self.framebuffer);
        logical_device.destroy_image_view(self.image_view);
        logical_device.destroy_image_view(self.depth_image_view);
        logical_device.destroy_image(self.image);
        logical_device.destroy_image(self.depth_image);
        logical_device.destroy_swapchain(self.swapchain);

        instance.destroy_surface(self.surface);
    }
}
