#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!("vez.rs");

use std::ffi::{CStr,CString};
use std::mem::MaybeUninit;
use std::os::raw::c_char;
use std::ops::BitOr;

//Helper function to convert a slice of Strings to a Vec of CStrings
fn to_c_string_array(in_array : &[String]) -> Vec<CString> {
    in_array
        .iter()
        .map(|s| CString::new(s.clone()).expect("failed to create CString"))
        .collect()
}

macro_rules! option_to_ptr {
    ( $option:ident ) => {
        match &$option {
            Some(value) => value.as_ptr(),
            None => std::ptr::null(),
        }
    };
}

pub struct ExtensionProperties {
    pub name : String,
    pub spec_version : u32,
}

pub struct LayerProperties {
    pub name : String,
    pub spec_version : u32,
    pub implementation_version : u32,
    pub description : String,
}

pub struct InstanceBuilder {
    enabled_layers : Vec<String>,
    enabled_extensions  : Vec<String>,
}

impl InstanceBuilder {
    pub fn new() -> InstanceBuilder {
        InstanceBuilder {
            enabled_layers : Vec::new(),
            enabled_extensions : Vec::new(),
        }
    }

    pub fn enumerate_extension_properties(&self, layer_name : Option<&str>) -> Vec<ExtensionProperties> {
        let raw_layer_name = match layer_name {
            Some(layer_name) => Some(CString::new(layer_name).unwrap()),
            None => None,
        };
        let p_layer_name = match raw_layer_name {
            Some(raw_layer_name) => raw_layer_name.as_ptr(),
            None => std::ptr::null(),
        };

        unsafe {
            let mut property_count = 0;
            vezEnumerateInstanceExtensionProperties(p_layer_name, &mut property_count, std::ptr::null_mut());

            let mut raw_properties = vec![std::mem::zeroed(); property_count as usize];
            vezEnumerateInstanceExtensionProperties(p_layer_name, &mut property_count, raw_properties.as_mut_ptr());

            if raw_properties.len() > 0 {
                return raw_properties.iter_mut().map(|raw| ExtensionProperties {
                    name : CStr::from_ptr(raw.extensionName.as_ptr()).to_owned().into_string().expect("failed to convert string"),
                    spec_version : raw.specVersion,
                }).collect();
            }
        }

        Vec::new()
    }

    pub fn enumerate_layer_properties(&self) -> Vec<LayerProperties> {
        unsafe {
            let mut property_count = 0;
            vezEnumerateInstanceLayerProperties(&mut property_count, std::ptr::null_mut());

            let mut raw_properties = vec![std::mem::zeroed(); property_count as usize];
            vezEnumerateInstanceLayerProperties(&mut property_count, raw_properties.as_mut_ptr());

            if raw_properties.len() > 0 {
                return raw_properties.iter_mut().map(|raw| LayerProperties {
                    name : CStr::from_ptr(raw.layerName.as_ptr()).to_owned().into_string().expect("failed to convert string"),
                    spec_version : raw.specVersion,
                    implementation_version : raw.implementationVersion,
                    description : CStr::from_ptr(raw.description.as_ptr()).to_owned().into_string().expect("failed to convert string")
                }).collect();
            }
        }
        
        Vec::new()
    }

    pub fn with_layers(&mut self, layers: &[String]) -> &mut InstanceBuilder {
        self.enabled_layers = layers.to_vec();
        self
    }

    pub fn with_extensions(&mut self, extensions: &[String]) -> &mut InstanceBuilder {
        self.enabled_extensions = extensions.to_vec();
        self
    }

    pub fn build(&self) -> Result<Instance, VkResult> {
        let app_info = VezApplicationInfo {
            pNext : std::ptr::null(),
            pApplicationName : CString::new("vez_app").unwrap().as_ptr(),
            applicationVersion : 1,
            pEngineName : CString::new("vez").unwrap().as_ptr(),
            engineVersion : 1,
        };

        let layers = to_c_string_array(&self.enabled_layers);

        let layers_c : Vec<*const c_char> = layers.iter().map(|s| s.as_ptr()).collect();

        let extensions = to_c_string_array(&self.enabled_extensions);

        let extensions_c : Vec<*const c_char> = extensions.iter().map(|s| s.as_ptr()).collect();

        let create_info = VezInstanceCreateInfo {
            pNext : std::ptr::null(),
            pApplicationInfo : &app_info,
            enabledLayerCount : layers_c.len() as u32,
            ppEnabledLayerNames : layers_c.as_ptr(),
            enabledExtensionCount : extensions_c.len() as u32,
            ppEnabledExtensionNames : extensions_c.as_ptr(),
        };

        unsafe {
            let mut instance : MaybeUninit<VkInstance> = MaybeUninit::zeroed();
            let result = vezCreateInstance(&create_info, instance.as_mut_ptr());
            
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }

            Ok(Instance {
                raw : instance.assume_init()
            })
        }
    }
}

pub struct Instance {
    pub raw : VkInstance,
}

impl Instance {
    pub fn destroy(self) {
        unsafe {
            vezDestroyInstance(self.raw);
        }
    }

    pub fn destroy_surface(&self, surface : VkSurfaceKHR) {
        unsafe {
            vkDestroySurfaceKHR(self.raw, surface, std::ptr::null());
        }
    }

    pub fn enumerate_physical_devices(&self) -> Result<Vec<PhysicalDevice>, VkResult> {
        unsafe {
            let mut physical_device_count = 0;
            let result = vezEnumeratePhysicalDevices(self.raw, &mut physical_device_count, std::ptr::null_mut());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }

            let mut raw_physical_devices = vec![std::ptr::null_mut(); physical_device_count as usize];
            let result = vezEnumeratePhysicalDevices(self.raw, &mut physical_device_count, raw_physical_devices.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }

            if raw_physical_devices.len() > 0 {
                return Ok(raw_physical_devices.iter_mut().map(|raw| PhysicalDevice { raw : *raw }).collect());
            }
        }

        Ok(Vec::new())
    }
}

pub struct PhysicalDevice {
    raw : VkPhysicalDevice,
}

impl PhysicalDevice {
    
    //TODO: get_properties
    //TODO: get_features
    //TODO: get_format_properties
    //TODO: get_image_format_properties
    //TODO: get_queue_family_properties
    //TODO: get_surface_formats
    
    pub fn get_present_support(&self, queue_family_index : u32, surface : &VkSurfaceKHR) -> bool {
        let mut supported = 0;
        let result = unsafe {
            vezGetPhysicalDevicePresentSupport(self.raw, queue_family_index, *surface, &mut supported)
        };

        if result == VkResult::VK_SUCCESS {
            return supported != 0;
        }

        false
    }

    pub fn enumerate_extension_properties(&self, layer_name : Option<&str>) -> Vec<ExtensionProperties> {
        let raw_layer_name = match layer_name {
            Some(layer_name) => Some(CString::new(layer_name).unwrap()),
            None => None,
        };

        let p_layer_name = match raw_layer_name {
            Some(raw_layer_name) => raw_layer_name.as_ptr(),
            None => std::ptr::null(),
        };

        unsafe {
            let mut property_count = 0;
            vezEnumerateDeviceExtensionProperties(self.raw, p_layer_name, &mut property_count, std::ptr::null_mut());

            let mut raw_properties = vec![std::mem::zeroed(); property_count as usize];
            vezEnumerateDeviceExtensionProperties(self.raw, p_layer_name, &mut property_count, raw_properties.as_mut_ptr());

            if raw_properties.len() > 0 {
                return raw_properties.iter_mut().map(|raw| ExtensionProperties {
                    name : CStr::from_ptr(raw.extensionName.as_ptr()).to_owned().into_string().expect("failed to convert string"),
                    spec_version : raw.specVersion,
                }).collect();
            }
        }

        Vec::new()
    }

    pub fn enumerate_layer_properties(&self) -> Vec<LayerProperties> {
        unsafe {
            let mut property_count = 0;
            vezEnumerateDeviceLayerProperties(self.raw, &mut property_count, std::ptr::null_mut());

            let mut raw_properties = vec![std::mem::zeroed(); property_count as usize];
            vezEnumerateDeviceLayerProperties(self.raw, &mut property_count, raw_properties.as_mut_ptr());

            if raw_properties.len() > 0 {
                return raw_properties.iter_mut().map(|raw| LayerProperties {
                    name : CStr::from_ptr(raw.layerName.as_ptr()).to_owned().into_string().expect("failed to convert string"),
                    spec_version : raw.specVersion,
                    implementation_version : raw.implementationVersion,
                    description : CStr::from_ptr(raw.description.as_ptr()).to_owned().into_string().expect("failed to convert string")
                }).collect();
            }
        }
        
        Vec::new()
    }

    pub fn logical_device_builder(&self) -> LogicalDeviceBuilder {
        LogicalDeviceBuilder {
            physical_device : self.raw,
            enabled_layers : Vec::new(),
            enabled_extensions : Vec::new(),
        }
    }
}

pub struct LogicalDeviceBuilder {
    physical_device : VkPhysicalDevice,
    enabled_layers : Vec<String>,
    enabled_extensions : Vec<String>,
}

impl LogicalDeviceBuilder {
    pub fn with_layers(&mut self, layers: &[String]) -> &mut LogicalDeviceBuilder {
        self.enabled_layers = layers.to_vec();
        self
    }

    pub fn with_extensions(&mut self, extensions: &[String]) -> &mut LogicalDeviceBuilder {
        self.enabled_extensions = extensions.to_vec();
        self
    }

    pub fn build(&mut self) -> Result<LogicalDevice,VkResult> {
        
        let layers = to_c_string_array(&self.enabled_layers);

        let layers_c : Vec<*const c_char> = layers.iter().map(|s| s.as_ptr()).collect();

        let extensions = to_c_string_array(&self.enabled_extensions);

        let extensions_c : Vec<*const c_char> = extensions.iter().map(|s| s.as_ptr()).collect();

        let create_info = VezDeviceCreateInfo {
            pNext : std::ptr::null(),
            enabledLayerCount : layers_c.len() as u32,
            ppEnabledLayerNames : layers_c.as_ptr(),
            enabledExtensionCount : extensions_c.len() as u32,
            ppEnabledExtensionNames : extensions_c.as_ptr(),
        };

        unsafe {
            let mut logical_device : MaybeUninit<VkDevice> = MaybeUninit::zeroed();
            let result = vezCreateDevice(self.physical_device, &create_info, logical_device.as_mut_ptr());
            
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }

            Ok(LogicalDevice {
                raw : logical_device.assume_init()
            })
        }
    }
}

pub struct LogicalDevice {
    raw : VkDevice,
}

impl LogicalDevice {
    pub fn destroy(self) {
        unsafe {
            vezDestroyDevice(self.raw);
        }
    }

    pub fn wait_idle(&self) {
        unsafe {
            vezDeviceWaitIdle(self.raw);
        }
    }
    
    pub fn get_queue(&self, queue_family_index : u32, queue_index : u32) -> Queue {
        
        let queue = unsafe {
            let mut queue = MaybeUninit::zeroed();
            vezGetDeviceQueue(self.raw, queue_family_index, queue_index, queue.as_mut_ptr());
            queue.assume_init()
        };

        Queue {
            raw: queue,
        }
    }

    fn get_typed_queue_helper(&self, get_queue_function : unsafe extern "C" fn(VkDevice, u32, *mut VkQueue), queue_index : u32) -> Queue {
        let queue = unsafe {
            let mut queue = MaybeUninit::zeroed();
            get_queue_function(self.raw, queue_index, queue.as_mut_ptr());
            queue.assume_init()
        };

        Queue {
            raw: queue,
        }
    }

    pub fn get_graphics_queue(&self, queue_index : u32) -> Queue {
        self.get_typed_queue_helper(vezGetDeviceGraphicsQueue, queue_index)
    }

    pub fn get_compute_queue(&self, queue_index : u32) -> Queue {
        self.get_typed_queue_helper(vezGetDeviceComputeQueue, queue_index)
    }
    
    pub fn get_transfer_queue(&self, queue_index : u32) -> Queue {
        self.get_typed_queue_helper(vezGetDeviceTransferQueue, queue_index)
    }

    pub fn create_swapchain(&self, surface : &VkSurfaceKHR, format : VkFormat, color_space : VkColorSpaceKHR, triple_buffer : bool) -> Result<Swapchain,VkResult> {
        let create_info = VezSwapchainCreateInfo {
            pNext : std::ptr::null(),
            surface : *surface,
            format : VkSurfaceFormatKHR {
                format,
                colorSpace : color_space,
            }, 
            tripleBuffer : if triple_buffer { VK_TRUE } else { VK_FALSE },
        };

        let raw_swapchain = unsafe {
            let mut swapchain = MaybeUninit::zeroed();
            let result = vezCreateSwapchain(self.raw, &create_info, swapchain.as_mut_ptr());

            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }

            swapchain.assume_init()
        };

        Ok(Swapchain {
            raw : raw_swapchain, 
        })
    }

    pub fn destroy_swapchain(&self, swapchain : Swapchain) {
        unsafe {
            vezDestroySwapchain(self.raw, swapchain.raw);
        }
    }
    
    //TODO: create_query_pool (Vez Version Not Exported)
    //TODO: destroy_query_pool (Vez version Not Exported)
    //TODO: get_query_pool_results (Vez Version Not Exported)

    pub fn create_shader_module(&self, shader_stage : ShaderStage, code : ShaderCode, entry_point : &str) -> Result<ShaderModule, String> {
        
        let (code_size, glsl_source, code) = match code {
            ShaderCode::GLSL(glsl) => (glsl.len(), Some(CString::new(glsl.clone()).expect("failed to create CString")), None),
            ShaderCode::SPIRV(spirv) => (spirv.len(), None, Some(spirv)),
        };

        let entry_point_c = CString::new(entry_point).expect("failed to create CString");
        
        let create_info = VezShaderModuleCreateInfo {
            pNext : std::ptr::null(),
            stage : shader_stage.0,
            codeSize : code_size,
            pCode : option_to_ptr!(code),
            pGLSLSource : option_to_ptr!(glsl_source),
            pEntryPoint : entry_point_c.as_ptr(),
        };
        
        let raw = unsafe {
            let mut raw_shader_module = MaybeUninit::zeroed();
            let result = vezCreateShaderModule(self.raw, &create_info, raw_shader_module.as_mut_ptr());

            if result != VkResult::VK_SUCCESS {
                if raw_shader_module.as_ptr() != std::ptr::null() {
                    let temp_shader_module = ShaderModule {
                        raw : raw_shader_module.assume_init(),
                    };

                    let info_log_string = temp_shader_module.get_info_log();

                    self.destroy_shader_module(temp_shader_module);

                    return Err(info_log_string);
                }

                return Err(format!("NoInfoLog: VkResult: {:?}", result));
            }
            
            raw_shader_module.assume_init()
        };
       
        Ok(ShaderModule {
            raw,
        })
    }

    pub fn destroy_shader_module(&self, shader_module : ShaderModule) {
        unsafe {
            vezDestroyShaderModule(self.raw, shader_module.raw);
        }
    }

    pub fn create_graphics_pipeline<'a>(&self, shader_modules: &[&'a ShaderModule]) -> Result<Pipeline<'a>,VkResult> {
        let shader_stage_create_infos : Vec<VezPipelineShaderStageCreateInfo> 
            = shader_modules.iter().map(|module| {
                VezPipelineShaderStageCreateInfo {
                    pNext : std::ptr::null(),
                    module : module.raw,
                    pEntryPoint : std::ptr::null(), //TODO: redefinable entry point
                    pSpecializationInfo : std::ptr::null(), //TODO: specialization info
                }
            }).collect();

        let graphics_pipeline_create_info = VezGraphicsPipelineCreateInfo {
            pNext : std::ptr::null(),
            stageCount : shader_stage_create_infos.len() as u32,
            pStages : shader_stage_create_infos.as_ptr(),
        };

        let raw = unsafe {
            let mut pipeline = MaybeUninit::zeroed();
            let result = vezCreateGraphicsPipeline(self.raw, &graphics_pipeline_create_info, pipeline.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            pipeline.assume_init()
        };

        Ok(Pipeline {
            raw,
            shader_modules : shader_modules.to_vec(),
        })
    }

    pub fn create_compute_pipeline<'a>(&self, shader_module : &'a ShaderModule) -> Result<Pipeline<'a>,VkResult> {
        let shader_stage_create_info = VezPipelineShaderStageCreateInfo {
            pNext : std::ptr::null(),
            module : shader_module.raw,
            pEntryPoint : std::ptr::null(), //TODO: redefinable entry point
            pSpecializationInfo : std::ptr::null(), //TODO: specialization info
        };

        let compute_pipeline_create_info = VezComputePipelineCreateInfo {
            pNext : std::ptr::null(),
            pStage : &shader_stage_create_info,
        };

        let raw = unsafe {
            let mut pipeline = MaybeUninit::zeroed();
            let result = vezCreateComputePipeline(self.raw, &compute_pipeline_create_info, pipeline.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            pipeline.assume_init()
        };

        Ok(Pipeline {
            raw,
            shader_modules : vec![shader_module],
        })
    }

    pub fn destroy_pipeline(&self, pipeline : Pipeline) {
        unsafe {
            vezDestroyPipeline(self.raw, pipeline.raw);
        }
    }

    pub fn create_vertex_input_format(&self, bindings : &[VkVertexInputBindingDescription], attributes : &[VkVertexInputAttributeDescription]) -> Result<VertexInputFormat,VkResult> {
        let create_info = VezVertexInputFormatCreateInfo {
            vertexBindingDescriptionCount : bindings.len() as u32,
            pVertexBindingDescriptions : if !bindings.is_empty() { bindings.as_ptr() } else { std::ptr::null() },
            vertexAttributeDescriptionCount : attributes.len() as u32,
            pVertexAttributeDescriptions : if !attributes.is_empty() { attributes.as_ptr() } else { std::ptr::null() },
        };

        let raw = unsafe {
            let mut vertex_input_format = MaybeUninit::zeroed();
            let result = vezCreateVertexInputFormat(self.raw, &create_info, vertex_input_format.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            vertex_input_format.assume_init()
        };

        Ok(VertexInputFormat {
            raw,
        })
    }
    
    pub fn destroy_vertex_input_format(&self, vertex_input_format : VertexInputFormat) {
        unsafe {
            vezDestroyVertexInputFormat(self.raw, vertex_input_format.raw);
        }
    }

    //TODO: create_sampler

    pub fn destroy_sampler(&self, sampler : Sampler) {
        unsafe {
            vezDestroySampler(self.raw, sampler.raw);
        }
    }

    pub fn create_buffer(&self, size : VkDeviceSize, memory_flags : MemoryUsage, usage : BufferUsage) -> Result<Buffer,VkResult> {
        let create_info = VezBufferCreateInfo {
            pNext : std::ptr::null(),
            size,
            usage : usage.0,
            queueFamilyIndexCount : 0, //TODO:
            pQueueFamilyIndices : std::ptr::null(), //TODO:
        };

        let raw = unsafe {
            let mut buffer = MaybeUninit::zeroed();
            let result = vezCreateBuffer(self.raw, memory_flags.0, &create_info, buffer.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            buffer.assume_init()
        };

        Ok(Buffer {
            raw
        })
    }

    pub fn destroy_buffer(&self, buffer : Buffer) {
        unsafe {
            vezDestroyBuffer(self.raw, buffer.raw);
        }
    }

    pub fn buffer_sub_data<T>(&self, buffer : &Buffer, data :&[T], offset : VkDeviceSize) -> VkResult {
        let size = (std::mem::size_of::<T>() * data.len()) as VkDeviceSize;
        unsafe {
            //TODO: ensure data will fit based on size and offset
            vezBufferSubData(self.raw, buffer.raw, offset, size, data.as_ptr() as *const std::ffi::c_void)
        }
    }

    //TODO: map_buffer
    //TODO: unmap_buffer
    //TODO: flush_mapped_buffer_ranges
    //TODO: invalidate mapped buffer ranges
    
    pub fn create_buffer_view(&self, buffer : &Buffer, format : VkFormat, range : std::ops::Range<VkDeviceSize>) -> Result<BufferView,VkResult> {
        let create_info = VezBufferViewCreateInfo {
            pNext : std::ptr::null(),
            buffer : buffer.raw,
            format,
            offset : range.start,
            range  : range.end,
        };

        let raw = unsafe {
            let mut view = MaybeUninit::zeroed();
            let result = vezCreateBufferView(self.raw, &create_info, view.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            view.assume_init()
        };

        Ok(BufferView {
            raw
        })
    }
    
    pub fn destroy_buffer_view(&self, view : BufferView) {
        unsafe {
            vezDestroyBufferView(self.raw, view.raw);
        }
    }

    pub fn create_image(&self, width: u32, height : u32, format : VkFormat, memory_flags : MemoryUsage, usage : ImageUsage) -> Result<Image,VkResult> {
        let create_info = VezImageCreateInfo {
            pNext : std::ptr::null(),
            flags : 0, //TODO:
            imageType : VkImageType::VK_IMAGE_TYPE_2D, //TODO:
            format,
            extent : VkExtent3D{ width, height, depth: 1 /*TODO:*/ },
            mipLevels : 1, //TODO:
            arrayLayers : 1, //TODO:
            samples : VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT, //TODO:
            tiling : VkImageTiling::VK_IMAGE_TILING_OPTIMAL, //TODO:
            usage : usage.0,
            queueFamilyIndexCount : 0,
            pQueueFamilyIndices : std::ptr::null(),
        };

        let raw = unsafe {
            let mut image = MaybeUninit::zeroed();
            let result = vezCreateImage(self.raw, memory_flags.0, &create_info, image.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            image.assume_init()
        };

        Ok(Image {
            raw
        })
    }
    
    pub fn destroy_image(&self, image : Image) {
        unsafe {
            vezDestroyImage(self.raw, image.raw);
        }
    }

    //TODO: image_sub_data

    pub fn create_image_view(&self, image : &Image, view_type : VkImageViewType, format : VkFormat) -> Result<ImageView,VkResult> {
        let create_info = VezImageViewCreateInfo {
            pNext: std::ptr::null(),
            image : image.raw,
            viewType : view_type,
            format,
            components : VkComponentMapping { //TODO:
                r : VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
                g : VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
                b : VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
                a : VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            subresourceRange : VezImageSubresourceRange { //TODO:
                baseMipLevel : 0,
                levelCount : 1,
                baseArrayLayer : 0,
                layerCount : 1,
            }
        };

        let raw = unsafe {
            let mut image_view = MaybeUninit::zeroed();
            let result = vezCreateImageView(self.raw, &create_info, image_view.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            image_view.assume_init()
        };

        Ok(ImageView {
            raw
        })
    }
    
    pub fn destroy_image_view(&self, image_view : ImageView) {
        unsafe {
            vezDestroyImageView(self.raw, image_view.raw);
        }
    }

    pub fn create_framebuffer(&self, attachments : &[&ImageView], width : u32, height : u32, layers : u32) -> Result<Framebuffer,VkResult> {
        
        let p_attachments : Vec<VkImageView> = attachments.iter().map(|attachment| attachment.raw).collect();
        
        let create_info = VezFramebufferCreateInfo {
            pNext : std::ptr::null(),
            attachmentCount : p_attachments.len() as u32,
            pAttachments : p_attachments.as_ptr(),
            width,
            height,
            layers,
        };

        let raw = unsafe {
            let mut framebuffer = MaybeUninit::zeroed();
            let result = vezCreateFramebuffer(self.raw, &create_info, framebuffer.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            framebuffer.assume_init()
        };

        Ok(Framebuffer {
            raw
        })
    }
    
    pub fn destroy_framebuffer(&self, framebuffer : Framebuffer) {
        unsafe {
            vezDestroyFramebuffer(self.raw, framebuffer.raw);
        }
    }

    pub fn allocate_command_buffers(&self, queue : &Queue, count : u32) -> Result<Vec<CommandBuffer>,VkResult> {
        let allocate_info = VezCommandBufferAllocateInfo {
            pNext : std::ptr::null(),
            queue : queue.raw,
            commandBufferCount : count,
        };

        let command_buffers = unsafe {
            let mut p_command_buffers = std::ptr::null_mut();
            let result = vezAllocateCommandBuffers(self.raw, &allocate_info, &mut p_command_buffers);
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            let mut raw_slice = std::slice::from_raw_parts_mut(p_command_buffers, count as usize);
            raw_slice
                .iter_mut()
                .map(|raw| CommandBuffer {
                    raw : raw
                })
                .collect()
        };

        Ok(command_buffers)
    }

    pub fn free_command_buffers(&self, command_buffers : &[CommandBuffer]) {
        let raw_command_buffers : Vec<VkCommandBuffer> = command_buffers.iter().map(|command_buffer| command_buffer.raw).collect();
        unsafe {
            vezFreeCommandBuffers(self.raw, command_buffers.len() as u32, raw_command_buffers.as_ptr());
        }
    }
}

pub struct Swapchain {
    raw : VezSwapchain,
}

impl Swapchain {
    pub fn get_surface_format(&self) -> VkSurfaceFormatKHR {
        unsafe {
            let mut surface_format = MaybeUninit::zeroed();
            vezGetSwapchainSurfaceFormat(self.raw, surface_format.as_mut_ptr());
            surface_format.assume_init()
        }
    }

    pub fn set_vsync_enabled(&self, enabled : bool) -> VkResult {
        unsafe {
            vezSwapchainSetVSync(self.raw, enabled as u32)
        }
    }
}

pub struct Queue {
    raw : VkQueue,
}

impl Queue {

    //Queue Submission: Submit n-command buffers, returning n-signal semaphores and 1 fence

    pub fn submit(&self, queue_submissions : &[QueueSubmission]) -> Result<Fence,VkResult> {
        
        struct RawSubmissionData {
            raw_command_buffers : Vec<VkCommandBuffer>, 
            raw_signal_semaphores : Vec<VkSemaphore>,
        }

        let mut raw_submissions_data : Vec<RawSubmissionData> = queue_submissions.iter().map(|submission| {
            let raw_command_buffers : Vec<VkCommandBuffer> = submission.command_buffers.iter().map(|command_buffer| command_buffer.raw).collect();
            let mut raw_signal_semaphores : Vec<VkSemaphore> = raw_command_buffers.iter().map(|_| std::ptr::null_mut()).collect();
            RawSubmissionData {
                raw_command_buffers,
                raw_signal_semaphores,
            }
        }).collect();

        let raw_submit_infos : Vec<VezSubmitInfo> = raw_submissions_data.iter_mut().map(|submit_data| {
            VezSubmitInfo {
                pNext : std::ptr::null(),
                waitSemaphoreCount : 0, //TODO:
                pWaitSemaphores : std::ptr::null(), //TODO:
                pWaitDstStageMask : std::ptr::null(), //TODO:
                commandBufferCount : submit_data.raw_command_buffers.len() as u32,
                pCommandBuffers : submit_data.raw_command_buffers.as_ptr(),
                signalSemaphoreCount : submit_data.raw_signal_semaphores.len() as u32,
                pSignalSemaphores : submit_data.raw_signal_semaphores.as_mut_ptr(),
            }
        }).collect();

        let raw = unsafe {
            let mut fence = MaybeUninit::zeroed();
            let result = vezQueueSubmit(self.raw, raw_submit_infos.len() as u32, raw_submit_infos.as_ptr(), fence.as_mut_ptr());
            if result != VkResult::VK_SUCCESS {
                return Err(result);
            }
            fence.assume_init()
        };

        Ok(Fence {
            raw
        })
    }
    
    pub fn present(&self, swapchains_and_images : &[(&Swapchain, &Image)]) -> VkResult {
        
        let raw_swapchains : Vec<VezSwapchain> = swapchains_and_images.iter().map(|(swapchain, _)| swapchain.raw).collect();
        let raw_images : Vec<VkImage> = swapchains_and_images.iter().map(|(_, image)| image.raw).collect();

        let mut raw_signal_semaphores : Vec<VkSemaphore> = swapchains_and_images.iter().map(|_| std::ptr::null_mut()).collect();
        let mut raw_results : Vec<VkResult> = swapchains_and_images.iter().map(|_| VkResult::VK_SUCCESS).collect();

        let present_info = VezPresentInfo {
            pNext : std::ptr::null(),
            waitSemaphoreCount : 0, //TODO:
            pWaitSemaphores : std::ptr::null(), //TODO:
            pWaitDstStageMask : std::ptr::null(), //TODO:
            swapchainCount : raw_swapchains.len() as u32,
            pSwapchains : raw_swapchains.as_ptr(),
            pImages : raw_images.as_ptr(),
            signalSemaphoreCount : 0, //TODO:
            pSignalSemaphores : std::ptr::null_mut(), //TODO:
            pResults : std::ptr::null_mut(), //TODO:
        };

        unsafe {
            let result = vezQueuePresent(self.raw, &present_info);
            result
        }
    }
    
    pub fn wait_idle(&self) -> VkResult {
        unsafe {
            vkQueueWaitIdle(self.raw)
        }
    }
}

pub struct QueueSubmission<'a> {
   pub command_buffers : &'a[&'a CommandBuffer],
   //TODO: wait Semaphores
   //TODO: wait DstStageMask
   //TODO: Signal Semaphores
}

pub struct Fence {
    raw : VkFence,
}

pub struct Semaphore {
    raw : VkSemaphore,
}

pub struct Event {
    raw : VkEvent,
}

pub enum ShaderCode {
    GLSL(String),
    SPIRV(Vec<u32>)
}

pub struct ShaderModule {
    raw : VkShaderModule,
}

impl ShaderModule {
    pub fn get_info_log(&self) -> String {
        unsafe {
            let mut info_log_length = 0;
            vezGetShaderModuleInfoLog(self.raw, &mut info_log_length, std::ptr::null_mut());

            let mut info_log = vec![0; info_log_length as usize];
            vezGetShaderModuleInfoLog(self.raw, &mut info_log_length, info_log.as_mut_ptr());

            CStr::from_ptr(info_log.as_ptr()).to_string_lossy().into_owned()
        }
    }

    pub fn get_binary(&self) -> Vec<u32> {
        unsafe {
            let mut binary_length = 0;
            vezGetShaderModuleBinary(self.raw, &mut binary_length, std::ptr::null_mut());

            let mut binary = vec![0u32; binary_length as usize];
            vezGetShaderModuleBinary(self.raw, &mut binary_length, binary.as_mut_ptr());

            binary
        }
    }
}

// enum PipelineResourceType {
//     Input(location),
//     Output(location),
//     Sampler,
//     CombinedImageSampler,
//     SampledImage,
//     StorageImage,
//     UniformTexelBuffer(members),
//     StorageTexelBuffer(members),
//     UniformBuffer(members),
//     StorageBuffer(members),
//     InputAttachment(subpass_index),
//     PushConstantBuffer,
// }

pub struct PipelineResource {
    pub set : u32,
    pub binding : u32,
   //TODO: Access (Read/Write)
   //TODO: Stages (Which Shader Stages use resource)
   //TODO: BaseType enum
   //TODO: ResourceType (enum above)

}

//Unlike in Vulkan, ShaderModules have to live as long as pipeline, store ShaderModule references in Pipeline
pub struct Pipeline<'a> {
    raw : VezPipeline,
    shader_modules : Vec<&'a ShaderModule>,
}

impl<'a> Pipeline<'a> {
    pub fn enumerate_pipeline_resources(&self) -> Vec<PipelineResource> {
       unimplemented!() //TODO:
    }

    pub fn get_resource(&self, name : &str) -> PipelineResource {
        unimplemented!() //TODO:
     }
}

pub struct VertexInputFormat {
    raw : VezVertexInputFormat,
}

pub struct Sampler {
    raw : VkSampler,
}

pub struct Buffer {
    raw : VkBuffer,
}

pub struct BufferView {
    raw : VkBufferView,
}

pub struct Image {
    raw : VkImage,
}

pub struct ImageView {
    raw : VkImageView,
}

pub struct Framebuffer {
    raw : VezFramebuffer,
}

pub struct CommandBuffer {
    raw : VkCommandBuffer,
}

impl CommandBuffer {

    pub fn begin_recording<F: Fn(&CommandRecorder)>(&self, usage : CommandBufferUsage, record_function : F ) {
        unsafe {
            vezBeginCommandBuffer(self.raw, usage.0);
        }

        let command_recorder = CommandRecorder {
            command_buffer : self,
        };

        record_function(&command_recorder);

        self.end_recording();
    }

    fn end_recording(&self) {
        unsafe {
            vezEndCommandBuffer();
        }
    }
    
    pub fn reset(&self) {
        unsafe {
            vezResetCommandBuffer(self.raw);
        }
    }
}

pub struct CommandRecorder<'a> {
    command_buffer : &'a CommandBuffer,
}

impl<'a> CommandRecorder<'a> {
    pub fn begin_render_pass<F: Fn()>(&self, framebuffer : &Framebuffer, attachments : &[VezAttachmentInfo], render_pass_function : F) {
        let begin_info = VezRenderPassBeginInfo {
            pNext: std::ptr::null(),
            framebuffer : framebuffer.raw,
            attachmentCount : attachments.len() as u32,
            pAttachments : if attachments.len() <= 0 { std::ptr::null() } else { attachments.as_ptr() },
        };

        unsafe {
            vezCmdBeginRenderPass(&begin_info);
        }

        render_pass_function();

        self.end_render_pass();
    }

    pub fn next_subpass(&self) {
        unsafe {
            vezCmdNextSubpass();
        }
    }
    
    fn end_render_pass(&self) {
        unsafe {
            vezCmdEndRenderPass();
        }
    }
    
    pub fn bind_pipeline(&self, pipeline : &Pipeline) {
        unsafe {
            vezCmdBindPipeline(pipeline.raw);
        }
    }

    pub fn push_constants<T>(&self, offset : u32, values : &[T]) {
        unsafe {
            vezCmdPushConstants(offset, (std::mem::size_of::<T>() * values.len()) as _, values.as_ptr() as _);
        }
    }

    pub fn bind_buffer(&self, buffer : &Buffer, offset : VkDeviceSize, range : VkDeviceSize, set : u32, binding : u32, array_element : u32) {
        unsafe {
            vezCmdBindBuffer(buffer.raw, offset, range, set, binding, array_element);
        }
    }

    pub fn bind_buffer_view(&self, buffer_view : &BufferView, set : u32, binding : u32, array_element : u32) {
        unsafe {
            vezCmdBindBufferView(buffer_view.raw, set, binding, array_element);
        }
    }

    pub fn bind_image_view(&self, image_view : &ImageView, sampler : &Sampler, set : u32, binding : u32, array_element : u32) {
        unsafe {
            vezCmdBindImageView(image_view.raw, sampler.raw, set, binding, array_element);
        }
    }

    pub fn bind_sampler(&self, sampler : &Sampler, set : u32, binding : u32, array_element : u32) {
        unsafe {
            vezCmdBindSampler(sampler.raw, set, binding, array_element);
        }
    }

    pub fn bind_vertex_buffer(&self, index : u32, vertex_buffer : &Buffer, offset : VkDeviceSize) {
        self.bind_vertex_buffers(index, &[(vertex_buffer, offset)]);
    }

    pub fn bind_vertex_buffers(&self, first_index : u32, vertex_buffers : &[(&Buffer, VkDeviceSize)]) {
        unsafe {
            let buffers : Vec<VkBuffer> = vertex_buffers.iter().map(|(buffer,_)| buffer.raw).collect();
            let offsets : Vec<VkDeviceSize> = vertex_buffers.iter().map(|(_, offset)| offset).cloned().collect();
            vezCmdBindVertexBuffers(first_index, buffers.len() as u32, buffers.as_ptr(), offsets.as_ptr());
        }
    }

    pub fn bind_index_buffer(&self, index_buffer : &Buffer, offset : VkDeviceSize, index_type : IndexType) {
        unsafe {
            vezCmdBindIndexBuffer(index_buffer.raw, offset, index_type.into());
        }
    }

    pub fn set_vertex_input_format(&self, vertex_input_format : &VertexInputFormat) {
        unsafe {
            vezCmdSetVertexInputFormat(vertex_input_format.raw);
        }
    }

    pub fn set_viewport(&self, index : u32, viewport : Viewport) {
        self.set_viewports(index, &[viewport]);
    }

    pub fn set_viewports(&self, first_viewport : u32, viewports : &[Viewport]) {
        unsafe {
            vezCmdSetViewport(first_viewport, viewports.len() as _, viewports.as_ptr() as _);
        }
    }

    pub fn set_scissor(&self, x : i32, y : i32, width : u32, height : u32) {
        unsafe {
            vezCmdSetScissor(0, 1, &VkRect2D {
                offset : VkOffset2D::new(x,y),
                extent : VkExtent2D::new(width, height),
            }); //TODO: multiple scissors
        }
    }

    pub fn set_viewport_state(&self, count : u32) {
        unsafe {
            vezCmdSetViewportState(count);
        }
    }
    
    pub fn set_input_assembly_state(&self, state : &VezInputAssemblyState) { //TODO: rustified structs
        unsafe {
            vezCmdSetInputAssemblyState(state);
        }
    }
    
    pub fn set_rasterization_state(&self, state : &VezRasterizationState) { //TODO: rustified structs
        unsafe {
            vezCmdSetRasterizationState(state);
        }
    }

    pub fn set_multisample_state(&self, state : &VezMultisampleState) { //TODO: rustified structs
        unsafe {
            vezCmdSetMultisampleState(state);
        }
    }
    
    pub fn set_depth_stencil_state(&self, state : &VezDepthStencilState) { //TODO: rustified structs
        unsafe {
            vezCmdSetDepthStencilState(state);
        }
    }

    pub fn set_color_blend_state(&self, state : &VezColorBlendState) { //TODO: rustified structs
        unsafe {
            vezCmdSetColorBlendState(state);
        }
    }

    pub fn set_light_width(&self, line_width: f32) {
        unsafe {
            vezCmdSetLineWidth(line_width);
        }
    }

    pub fn set_depth_bias(&self, constant: f32, clamp : f32, slope : f32) {
        unsafe {
            vezCmdSetDepthBias(constant, clamp, slope);
        }
    }

    pub fn set_blend_constants(&self, blend_constants : &[f32;4]) {
        unsafe {
            vezCmdSetBlendConstants(blend_constants.as_ptr());
        }
    }

    pub fn set_depth_bounds(&self, min : f32, max : f32) {
        unsafe {
            vezCmdSetDepthBounds(min, max);
        }
    }

    pub fn set_stencil_compare_mask(&self, face_mask : StencilFace, compare_mask : u32) {
        unsafe {
            vezCmdSetStencilCompareMask(face_mask.0, compare_mask);
        }
    }

    pub fn set_stencil_write_mask(&self, face_mask : StencilFace, write_mask : u32) {
        unsafe {
            vezCmdSetStencilWriteMask(face_mask.0, write_mask);
        }
    }

    pub fn set_stencil_reference(&self, face_mask : StencilFace, reference : u32) {
        unsafe {
            vezCmdSetStencilReference(face_mask.0, reference);
        }
    }

    pub fn draw(&self, vertices : std::ops::Range<u32>, instances : std::ops::Range<u32>) {
        let vertex_count = vertices.end - vertices.start;
        let instance_count = instances.end - instances.start;
        unsafe {
            vezCmdDraw(vertex_count, instance_count, vertices.start, instances.start);
        }
    }

    pub fn draw_indexed(&self, indices : std::ops::Range<u32>, instances : std::ops::Range<u32>, vertex_offset : i32) {
        let index_count = indices.end - indices.start;
        let instance_count = instances.end - instances.start;
        unsafe {
            vezCmdDrawIndexed(index_count, instance_count, indices.start, vertex_offset, instances.start);
        }
    }

    pub fn draw_indirect(&self, buffer : &Buffer, offset : VkDeviceSize, draw_count : u32, stride : u32) {
        unsafe {
            vezCmdDrawIndirect(buffer.raw, offset, draw_count, stride);
        }
    }

    pub fn draw_indexed_indirect(&self, buffer : &Buffer, offset : VkDeviceSize, draw_count : u32, stride : u32) {
        unsafe {
            vezCmdDrawIndexedIndirect(buffer.raw, offset, draw_count, stride);
        }
    }

    pub fn dispatch(&self, group_counts : (u32, u32, u32)) {
        unsafe {
            vezCmdDispatch(group_counts.0, group_counts.1, group_counts.2);
        }
    }
    
    pub fn dispatch_indirect(&self, buffer : &Buffer, offset : VkDeviceSize) {
        unsafe {
            vezCmdDispatchIndirect(buffer.raw, offset);
        }
    }

    //TODO: rustified structs / names for VezBufferCopy, VezImageCopy, VezImageBlit, VezBufferImageCopy, VezImageResolve?

    pub fn copy_buffer(&self, src : &Buffer, dst : &Buffer, regions : &[VezBufferCopy]) {
        unsafe {
            vezCmdCopyBuffer(src.raw, dst.raw, regions.len() as _, regions.as_ptr());
        }
    }

    pub fn copy_image(&self, src : &Image, dst : &Image, regions : &[VezImageCopy]) {
        unsafe {
            vezCmdCopyImage(src.raw, dst.raw, regions.len() as _, regions.as_ptr());
        }
    }

    pub fn blit_image(&self, src : &Image, dst : &Image, regions : &[VezImageBlit], filter : Filter) {
        unsafe {
            vezCmdBlitImage(src.raw, dst.raw, regions.len() as _, regions.as_ptr(), filter.into());
        }
    }
    
    pub fn copy_buffer_to_image(&self, src : &Buffer, dst : &Image, regions : &[VezBufferImageCopy]) {
        unsafe {
            vezCmdCopyBufferToImage(src.raw, dst.raw, regions.len() as _, regions.as_ptr());
        }
    }
    
    pub fn copy_image_to_buffer(&self, src : &Image, dst : &Buffer, regions : &[VezBufferImageCopy]) {
        unsafe {
            vezCmdCopyImageToBuffer(src.raw, dst.raw, regions.len() as _, regions.as_ptr());
        }
    }

    pub fn update_buffer<T>(&self, dst : &Buffer, offset : VkDeviceSize, data : &[T]) {
        unsafe {
            vezCmdUpdateBuffer(dst.raw, offset, (data.len() * std::mem::size_of::<T>()) as _, data.as_ptr() as *const std::ffi::c_void);
        }
    }
    
    pub fn fill_buffer(&self, dst : &Buffer, fill_region : std::ops::Range<VkDeviceSize>, value : u32) {
        unsafe {
            vezCmdFillBuffer(dst.raw, fill_region.start, fill_region.end - fill_region.start, value);
        }
    }

    pub fn clear_color_image(&self, image : &Image, clear_value : ClearColor, ranges : &[VezImageSubresourceRange]) {
        unsafe {
            vezCmdClearColorImage(image.raw, &clear_value.into(), ranges.len() as _, ranges.as_ptr());
        }
    }

    pub fn clear_depth_stencil_image(&self, image : &Image, clear_value : ClearDepthStencil, ranges : &[VezImageSubresourceRange]) {
        unsafe {
            vezCmdClearDepthStencilImage(image.raw, &clear_value.into(), ranges.len() as _, ranges.as_ptr());
        }
    }

    pub fn resolve_image(&self, src : &Image, dst : &Image, regions : &[VezImageResolve]) {
        unsafe {
            vezCmdResolveImage(src.raw, dst.raw, regions.len() as _, regions.as_ptr());
        }
    }

    pub fn clear_attachments(&self, attachments : &[VezClearAttachment], rects : &[VkClearRect]) {
        unsafe {
            vezCmdClearAttachments(attachments.len() as _, attachments.as_ptr(), rects.len() as _, rects.as_ptr());
        }
    }

    //FIXME: linker errors, need to export in V-EZ/Source/VEZ.def
    // pub fn set_event(&self, event : &Event, stage_mask : PipelineStage) {
    //     unsafe {
    //         vkCmdSetEvent(event.raw, stage_mask.0);
    //     }
    // }
    
    //FIXME: linker errors, need to export in V-EZ/Source/VEZ.def
    // pub fn reset_event(&self, event : &Event, stage_mask : PipelineStage) {
    //     unsafe {
    //         vkCmdResetEvent(event.raw, stage_mask.0);
    //     }
    // }
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryUsage(VezMemoryFlags);

impl MemoryUsage {
    pub const GPU_ONLY : MemoryUsage = MemoryUsage(VezMemoryFlagsBits::VEZ_MEMORY_GPU_ONLY as _);
    pub const CPU_ONLY : MemoryUsage = MemoryUsage(VezMemoryFlagsBits::VEZ_MEMORY_CPU_ONLY as _);
    pub const CPU_TO_GPU : MemoryUsage = MemoryUsage(VezMemoryFlagsBits::VEZ_MEMORY_CPU_TO_GPU as _);
    pub const GPU_TO_CPU : MemoryUsage = MemoryUsage(VezMemoryFlagsBits::VEZ_MEMORY_GPU_TO_CPU as _);
    pub const DEDICATED_ALLOCATION : MemoryUsage = MemoryUsage(VezMemoryFlagsBits::VEZ_MEMORY_DEDICATED_ALLOCATION as _);
    pub const NO_ALLOCATION : MemoryUsage = MemoryUsage(VezMemoryFlagsBits::VEZ_MEMORY_NO_ALLOCATION as _);
}

impl BitOr for MemoryUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        MemoryUsage(self.0 | rhs.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CommandBufferUsage(VkCommandBufferUsageFlags);

impl CommandBufferUsage {
    pub const ONE_TIME_SUBMIT : CommandBufferUsage = CommandBufferUsage(VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT as _);
    pub const RENDER_PASS_CONTINUE : CommandBufferUsage = CommandBufferUsage(VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT as _);
    pub const SIMULTANEOUS_USE : CommandBufferUsage = CommandBufferUsage(VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT as _);
}

impl BitOr for CommandBufferUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        CommandBufferUsage(self.0 | rhs.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct BufferUsage(VkBufferUsageFlags);

impl BufferUsage {
    pub const TRANSFER_SRC : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT as _);
    pub const TRANSFER_DST : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT as _);
    pub const UNIFORM_TEXEL_BUFFER : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT as _);
    pub const STORAGE_TEXEL_BUFFER : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT as _);
    pub const UNIFORM_BUFFER : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT as _);
    pub const STORAGE_BUFFER : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT as _);
    pub const INDEX_BUFFER : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_INDEX_BUFFER_BIT as _);
    pub const VERTEX_BUFFER : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as _);
    pub const INDIRECT_BUFFER : BufferUsage = BufferUsage(VkBufferUsageFlagBits::VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT as _);
}

impl BitOr for BufferUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        BufferUsage(self.0 | rhs.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ImageUsage(VkImageUsageFlags);

impl ImageUsage {
    pub const TRANSFER_SRC : ImageUsage = ImageUsage(VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_SRC_BIT as _);
    pub const TRANSFER_DST : ImageUsage = ImageUsage(VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT as _);
    pub const SAMPLED : ImageUsage = ImageUsage(VkImageUsageFlagBits::VK_IMAGE_USAGE_SAMPLED_BIT as _);
    pub const STORAGE : ImageUsage = ImageUsage(VkImageUsageFlagBits::VK_IMAGE_USAGE_STORAGE_BIT as _);
    pub const COLOR_ATTACHMENT : ImageUsage = ImageUsage(VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT as _);
    pub const DEPTH_STENCIL_ATTACHMENT : ImageUsage = ImageUsage(VkImageUsageFlagBits::VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT as _);
    pub const TRANSIENT_ATTACHMENT : ImageUsage = ImageUsage(VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT as _);
    pub const INPUT_ATTACHMENT : ImageUsage = ImageUsage(VkImageUsageFlagBits::VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT as _);
}

impl BitOr for ImageUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        ImageUsage(self.0 | rhs.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PipelineStage(VkPipelineStageFlags);

impl PipelineStage {
    pub const TOP_OF_PIPE : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT as _ );
    pub const DRAW_INDIRECT : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT as _ );
    pub const VERTEX_INPUT : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_VERTEX_INPUT_BIT as _ );
    pub const VERTEX_SHADER : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_VERTEX_SHADER_BIT as _ );
    pub const TESSELLATION_CONTROL_SHADER : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT as _ );
    pub const TESSELLATION_EVALUATION_SHADER : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT as _ );
    pub const GEOMETRY_SHADER : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT as _ );
    pub const FRAGMENT_SHADER : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT as _ );
    pub const EARLY_FRAGMENT_TESTS : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT as _ );
    pub const LATE_FRAGMENT_TESTS : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT as _ );
    pub const COLOR_ATTACHMENT_OUTPUT : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT as _);
    pub const COMPUTE_SHADER : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT as _ );
    pub const TRANSFER : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as _ );
    pub const BOTTOM_OF_PIPE : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT as _ );
    pub const HOST : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_HOST_BIT as _ );
    pub const ALL_GRAPHICS : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT as _ );
    pub const ALL_COMMANDS : PipelineStage = PipelineStage(VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT as _ );
}

impl BitOr for PipelineStage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        PipelineStage(self.0 | rhs.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct StencilFace(VkStencilFaceFlags);

impl StencilFace {
    pub const FRONT : StencilFace = StencilFace(VkStencilFaceFlagBits::VK_STENCIL_FACE_FRONT_BIT as _);
    pub const BACK : StencilFace = StencilFace(VkStencilFaceFlagBits::VK_STENCIL_FACE_BACK_BIT as _);
    pub const BOTH : StencilFace = StencilFace(VkStencilFaceFlagBits::VK_STENCIL_FACE_FRONT_AND_BACK as _);
}

impl BitOr for StencilFace {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        StencilFace(self.0 | rhs.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ShaderStage(VkShaderStageFlagBits);

impl ShaderStage {
    pub const VERTEX : ShaderStage = ShaderStage(VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT);
    pub const TESSELLATION_CONTROL : ShaderStage = ShaderStage(VkShaderStageFlagBits::VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    pub const TESSELLATION_EVALUATION : ShaderStage = ShaderStage(VkShaderStageFlagBits::VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    pub const GEOMETRY : ShaderStage = ShaderStage(VkShaderStageFlagBits::VK_SHADER_STAGE_GEOMETRY_BIT);
    pub const FRAGMENT : ShaderStage = ShaderStage(VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT);
    pub const COMPUTE : ShaderStage = ShaderStage(VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
    pub const ALL_GRAPHICS : ShaderStage = ShaderStage(VkShaderStageFlagBits::VK_SHADER_STAGE_ALL_GRAPHICS);
    pub const ALL : ShaderStage = ShaderStage(VkShaderStageFlagBits::VK_SHADER_STAGE_ALL);
}

//TODO: ShaderStages (BitOr of ShaderStage)

#[derive(Debug, Copy, Clone)]
pub enum ClearColor {
    Float([f32; 4usize]),
    Int([i32; 4usize]),
    UInt([u32; 4usize]),
}

impl From<ClearColor> for VkClearColorValue {
    fn from(clear_color: ClearColor) -> Self {
        match clear_color {
            ClearColor::Float(float_array) => VkClearColorValue { float32 : float_array},
            ClearColor::Int(int_array) => VkClearColorValue { int32 : int_array},
            ClearColor::UInt(unsigned_int_array) => VkClearColorValue { uint32 : unsigned_int_array},
        }
    }
}

pub type ClearDepthStencil = VkClearDepthStencilValue;

#[derive(Debug, Copy, Clone)]
pub enum ClearValue {
    Color(ClearColor),
    DepthStencil(ClearDepthStencil),
}

impl From<ClearValue> for VkClearValue {
    fn from(clear_value: ClearValue) -> Self {
        match clear_value {
            ClearValue::Color(clear_color) => VkClearValue { color : clear_color.into() },
            ClearValue::DepthStencil(clear_depth_stencil) => VkClearValue { depthStencil : clear_depth_stencil},
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum IndexType {
    U16,
    U32,
}

impl From<IndexType> for VkIndexType {
    fn from(index_type : IndexType) -> Self {
        match index_type {
            IndexType::U16 => VkIndexType::VK_INDEX_TYPE_UINT16,
            IndexType::U32 => VkIndexType::VK_INDEX_TYPE_UINT32,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Filter {
    Nearest,
    Linear,
    CubicImage,
}

impl From<Filter> for VkFilter {
    fn from(filter : Filter) -> Self {
        match filter {
            Filter::Nearest => VkFilter::VK_FILTER_NEAREST,
            Filter::Linear =>  VkFilter::VK_FILTER_LINEAR,
            Filter::CubicImage => VkFilter::VK_FILTER_CUBIC_IMG,
        }
    }
}

impl VkOffset2D {
    fn new(x: i32, y: i32) -> VkOffset2D {
        VkOffset2D {
            x,
            y
        }
    }
}

impl VkExtent2D {
    fn new(width: u32, height : u32) -> VkExtent2D {
        VkExtent2D {
            width,
            height,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Viewport {
    pub x : f32,
    pub y : f32,
    pub width : f32,
    pub height : f32,
    pub min_depth : f32,
    pub max_depth : f32,
}

impl From<Viewport> for VkViewport {
    fn from(viewport : Viewport) -> Self {
        VkViewport {
            x : viewport.x,
            y : viewport.y,
            width : viewport.width,
            height : viewport.height,
            minDepth : viewport.min_depth,
            maxDepth : viewport.max_depth,
        }
    }
}
