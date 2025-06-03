use std::{ffi::c_void, path::Path, ptr::null};

use image::{GenericImageView, ImageBuffer, Rgb, RgbImage, imageops::FilterType};
use ndarray::{Array, Axis, s};
use ort::{
	execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
	memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
	session::{RunOptions, Session},
	tensor::Shape,
	value::TensorRefMut,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 640;
const THRESHOLD: f32 = 0.05;

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
	x1: f32,
	y1: f32,
	x2: f32,
	y2: f32,
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
	(box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
	((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}

// const YOLOV8M_URL: &str = "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/yolov8m.onnx";

#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

use image::{Rgba, open};
use imageproc::drawing::draw_hollow_rect_mut;

extern crate cudarc;
extern crate nvidia_video_codec_sdk;

use cudarc::{
	driver::{
		CudaContext, DevicePtr, LaunchConfig, PushKernelArg,
		result::{malloc_async, module},
	},
	nvrtc::Ptx,
};
use ffmpeg::{codec::Id, software::scaler};
use nvidia_video_codec_sdk::Decoder;
use nvidia_video_codec_sdk::Dim;
use nvidia_video_codec_sdk::Frame;
use std::convert::TryFrom;
extern crate ffmpeg_next as ffmpeg;
use std::fs::File;
use std::path::PathBuf;
use std::str::FromStr;

use ffmpeg::bsfilter::BSFContext;
use ffmpeg::{Packet, format};

// fn extract_packets_mp4mpeg(mut ictx: format::context::Input, stream_id: i32, extra_data_size: usize, extra_data: *mut u8) -> VecDeque<Packet> {
// 	// let mut filter = filter.unwrap();
// 	let mut packets: VecDeque<Packet> = VecDeque::new();
// 	for p in ictx.packets() {
// 		if p.0.id() == stream_id {
// 			let packet = if packets.len() == 0 {
// 				let packet_size = extra_data_size as usize + p.1.size() - 3 * size_of::<u8>();
// 				let mut buffer = vec![0u8; packet_size];
// 				let ptr: *mut u8 = buffer.as_mut_ptr();
// 				unsafe {
// 					ptr::copy_nonoverlapping(extra_data, ptr, extra_data_size);
// 					ptr::copy_nonoverlapping(&p.1.data().unwrap()[3], ptr.add(extra_data_size), p.1.size() - 3);
// 					Packet::copy(std::slice::from_raw_parts_mut(ptr, extra_data_size + p.1.size() - 3))
// 				}
// 			} else {
// 				p.1
// 			};
// 			packets.push_back(packet);
// 		}
// 	}
// 	packets
// }

struct InitInfo {
	source_path: PathBuf,
	is_rtsp: bool,
	resize_info: Dim,
}

pub trait PacketProvider {
	fn next_packet(&mut self) -> Option<Packet>;
	fn codec_id(&self) -> Id;
	fn parameters(&self) -> ffmpeg::codec::Parameters;
}

struct PacketProviderFromFile {
	ictx: format::context::Input,
	paramemters: ffmpeg::codec::Parameters,
	codec_id: Id,
	stream_id: i32,
	filter: BSFContext,
}

impl PacketProviderFromFile {
	fn new(file_path: &PathBuf) -> Self {
		let ictx = format::input(file_path).unwrap();
		let stream: ffmpeg::Stream<'_> = ictx.streams().best(ffmpeg::media::Type::Video).unwrap();
		let paramemters: ffmpeg::codec::Parameters = stream.parameters();
		let stream_id: i32 = stream.id();
		let codec_id = stream.parameters().id();

		let filter = {
			let is_mp4h264 = codec_id == ffmpeg_next::codec::Id::H264;
			let is_mp4hevc = codec_id == ffmpeg_next::codec::Id::HEVC;
			if is_mp4h264 {
				BSFContext::new("h264_mp4toannexb", &stream.parameters()).unwrap()
			} else if is_mp4hevc {
				BSFContext::new("hevc_mp4toannexb", &stream.parameters()).unwrap()
			} else {
				panic!("Filter only done for h264 and hevc. Maybe add more");
			}
		};
		PacketProviderFromFile {
			ictx,
			paramemters,
			stream_id,
			filter,
			codec_id,
		}
	}
}
impl PacketProvider for PacketProviderFromFile {
	fn next_packet(&mut self) -> Option<Packet> {
		loop {
			if let Some(pack) = self.ictx.packets().next() {
				if pack.0.id() == self.stream_id {
					self.filter.send_packet(pack.1).unwrap();
					let packet = self.filter.receive_packet().unwrap();
					return Some(packet);
				}
			} else {
				return None;
			}
		}
	}

	fn codec_id(&self) -> Id {
		self.codec_id
	}

	fn parameters(&self) -> ffmpeg::codec::Parameters {
		self.paramemters.clone()
	}
}

struct PacketProviderFromRTSP {
	ictx: format::context::Input,
	paramemters: ffmpeg::codec::Parameters,
	codec_id: Id,
	stream_id: i32,
}
impl PacketProviderFromRTSP {
	fn new(file_path: &PathBuf) -> Self {
		let mut input_opts = ffmpeg::Dictionary::new();
		input_opts.set("rtsp_transport", "tcp"); // Reliable packet delivery
		input_opts.set("max_delay", "500000"); // (Optional) reduce latency
		let ictx = format::input_with_dictionary(file_path, input_opts).unwrap();
		let stream: ffmpeg::Stream<'_> = ictx.streams().best(ffmpeg::media::Type::Video).unwrap();
		let paramemters: ffmpeg::codec::Parameters = stream.parameters();
		let stream_id: i32 = stream.id();
		let codec_id = stream.parameters().id();
		PacketProviderFromRTSP {
			ictx,
			paramemters,
			codec_id,
			stream_id,
		}
	}
}
impl PacketProvider for PacketProviderFromRTSP {
	fn next_packet(&mut self) -> Option<Packet> {
		loop {
			if let Some(pack) = self.ictx.packets().next() {
				if pack.0.id() == self.stream_id {
					return Some(pack.1);
				}
			} else {
				return None;
			}
		}
	}

	fn codec_id(&self) -> Id {
		self.codec_id
	}

	fn parameters(&self) -> ffmpeg::codec::Parameters {
		self.paramemters.clone()
	}
}

pub struct CpuFrameIter {
	pending_frames: bool,
	decoder: ffmpeg::decoder::Video,
	scaler: ffmpeg::software::scaling::Context,
	packet_provider: Box<dyn PacketProvider>,
}

impl CpuFrameIter {
	fn convert_frame(&mut self, decoded: &ffmpeg::util::frame::Video) -> Frame {
		let mut buffer: Vec<f32> = vec![0f32; 3 * 640 * 640];
		let mut rgb_frame = ffmpeg::util::frame::Video::empty();
		self.scaler.run(decoded, &mut rgb_frame).expect("Scaling failed");

		let data = rgb_frame.data(0);
		let stride = rgb_frame.stride(0);

		for c in 0..3 {
			for y in 0..640 {
				for x in 0..640 {
					let src_index = y * stride + x * 3 + c;
					let dst_index = c * 640 * 640 + y * 640 + x;
					buffer[dst_index] = data[src_index] as f32 / 255.0;
				}
			}
		}

		// copy to GPU only
		let size: usize = 3 * 640 * 640 * 4;
		let mut dev_ptr: *mut c_void = std::ptr::null_mut();
		unsafe {
			cudarc::runtime::sys::cudaMalloc(&mut dev_ptr, size);
		}
		unsafe {
			let _res = cudarc::runtime::result::memcpy_htod_sync(dev_ptr, &buffer);
		};

		Frame::new(dev_ptr as u64, size)
	}
}

impl TryFrom<InitInfo> for CpuFrameIter {
	type Error = String;

	fn try_from(init: InitInfo) -> Result<Self, Self::Error> {
		let packet_provider: Box<dyn PacketProvider> = if init.is_rtsp {
			Box::new(PacketProviderFromRTSP::new(&init.source_path))
		} else {
			Box::new(PacketProviderFromFile::new(&init.source_path))
		};
		let context_decoder = ffmpeg::codec::context::Context::from_parameters(packet_provider.parameters()).unwrap();

		let decoder: ffmpeg::decoder::Video = context_decoder.decoder().video().unwrap();
		let scaler: ffmpeg::software::scaling::Context = ffmpeg::software::scaling::Context::get(
			decoder.format(),
			decoder.width(),
			decoder.height(),
			ffmpeg::format::Pixel::RGB24,
			init.resize_info.w as u32,
			init.resize_info.h as u32,
			ffmpeg::software::scaling::flag::Flags::BILINEAR,
		)
		.unwrap();

		Ok(Self {
			pending_frames: false,
			decoder,
			scaler,
			packet_provider,
		})
	}
}

impl Iterator for CpuFrameIter {
	type Item = Frame;

	fn next(&mut self) -> Option<Self::Item> {
		let mut decoded = ffmpeg::util::frame::Video::empty();

		// Step 1: If we know there's a pending frame, try to receive it
		if self.pending_frames {
			if self.decoder.receive_frame(&mut decoded).is_ok() {
				return Some(self.convert_frame(&decoded));
			} else {
				self.pending_frames = false;
			}
		}

		// Step 2: Send packets until we receive a frame
		loop {
			let packet = self.packet_provider.next_packet();
			if packet.is_some() {
				self.decoder.send_packet(&packet.unwrap()).unwrap();
				if self.decoder.receive_frame(&mut decoded).is_ok() {
					self.pending_frames = true;
					return Some(self.convert_frame(&decoded));
				} else {
					continue;
				}
			} else {
				return None;
			}
		}
	}
}

pub struct FrameIter {
	num_decoded_frames: usize,
	decoder: Decoder,
	packet_provider: Box<dyn PacketProvider>,
}

impl TryFrom<InitInfo> for FrameIter {
	type Error = String;

	fn try_from(init: InitInfo) -> Result<Self, Self::Error> {
		let ctx = CudaContext::new(0).unwrap();
		let cuda_stream = ctx.new_stream().unwrap();

		let packet_provider: Box<dyn PacketProvider> = if init.is_rtsp {
			Box::new(PacketProviderFromRTSP::new(&init.source_path))
		} else {
			Box::new(PacketProviderFromFile::new(&init.source_path))
		};
		let codec_id = ffmpeg_id_to_nv_id(packet_provider.codec_id());

		let decoder =
			Decoder::initialize_with_cuda(ctx, cuda_stream, codec_id, init.resize_info).expect("NVIDIA Video Codec SDK should be installed correctly.");
		Ok(Self {
			num_decoded_frames: 0,
			decoder,
			packet_provider,
		})
	}
}

impl Iterator for FrameIter {
	type Item = Frame;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			if self.num_decoded_frames == 0 {
				let mut packet = self.packet_provider.next_packet()?;
				let size = packet.size();
				let data = packet.data_mut().unwrap();
				self.num_decoded_frames = self.decoder.decode(data.as_mut_ptr(), size as u64);
				// println!("Got decoded frames = {}", self.num_decoded_frames);
			}

			if self.num_decoded_frames != 0 {
				let frame = self.decoder.get_frame();
				self.num_decoded_frames -= 1;
				return frame;
			}
		}
	}
}

fn save_rgb_as_image(h_rgb: &[f32], width: usize, height: usize, path: &str) {
	let mut img = RgbImage::new(width as u32, height as u32);

	let val: f32 = 255.0;
	for y in 0..height {
		for x in 0..width {
			let idx = (y * width + x) * 3;
			let r = (h_rgb[idx] * val).clamp(0.0, 255.0) as u8;
			let g = (h_rgb[idx + 1] * val).clamp(0.0, 255.0) as u8;
			let b = (h_rgb[idx + 2] * val).clamp(0.0, 255.0) as u8;
			img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
		}
	}

	img.save(path).unwrap();
	println!("Saved RGB image to {}", path);
}

// use cudarc::driver::CudaSlice;

fn save_chw_as_image(h_data: &[f32], width: usize, height: usize, path: &str) {
	// Copy CHW data from GPU to host
	// Create image buffer
	let mut img = RgbImage::new(width as u32, height as u32);

	// CHW layout:
	// [R R R... (H×W)] [G G G... (H×W)] [B B B... (H×W)]
	let r_channel = &h_data[0..width * height];
	let g_channel = &h_data[width * height..2 * width * height];
	let b_channel = &h_data[2 * width * height..3 * width * height];

	for y in 0..height {
		for x in 0..width {
			let idx = y * width + x;
			let r = (r_channel[idx].clamp(0.0, 1.0) * 255.0) as u8;
			let g = (g_channel[idx].clamp(0.0, 1.0) * 255.0) as u8;
			let b = (b_channel[idx].clamp(0.0, 1.0) * 255.0) as u8;

			img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
		}
	}

	img.save(path).unwrap();
}

// use std::path::Path;

fn draw_boxes_on_yuv(
	ptr: *mut c_void, // Now takes mutable reference
	width: usize,
	height: usize,
	boxes: &[(BoundingBox, &str, f32)],
	scale_x: f32,
	scale_y: f32,
) {
	unsafe {
		let y_plane_size = width * height;
		let total_size = y_plane_size + (width * height / 2);
		let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, total_size);

		let (y_plane, uv_plane) = slice.split_at_mut(y_plane_size);

		for (bbox, _, _) in boxes {
			let x1 = bbox.x1.round() as usize;
			let y1 = bbox.y1.round() as usize;
			let x2 = bbox.x2.round() as usize;
			let y2 = bbox.y2.round() as usize;

			// Draw horizontal lines
			for x in x1..=x2 {
				if x < width {
					if y1 < height {
						y_plane[y1 * width + x] = 255;
					}
					if y2 < height {
						y_plane[y2 * width + x] = 255;
					}
				}
			}

			// Draw vertical lines
			for y in y1..=y2 {
				if y < height {
					if x1 < width {
						y_plane[y * width + x1] = 255;
					}
					if x2 < width {
						y_plane[y * width + x2] = 255;
					}
				}
			}
		}
	}
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	// Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,ort=debug".into()))
		.with(tracing_subscriber::fmt::layer())
		.init();

	ort::init()
		.with_execution_providers([TensorRTExecutionProvider::default().with_engine_cache(true).build().error_on_failure()])
		.commit()?;

	let ctx = CudaContext::new(0)?;
	let stream = ctx.new_stream()?;

	let module = ctx.load_module(Ptx::from_file("/home/satyam/dev/ort/examples/cudarc/kernel.ptx"))?;
	let f = module.load_function("nv12_to_normalized_rgb_kernel").unwrap();
	let g = module.load_function("interleaved_to_chw_kernel").unwrap();
	let mut session: Session = Session::builder()?
		.with_intra_threads(2)?
		.with_log_level(ort::logging::LogLevel::Warning)?
		.commit_from_file("/home/satyam/models/yolov8s.onnx")?;

	let video: bool = true;
	let test_frame: bool = false;

	if video {
		let mut out_file = File::create("/home/satyam/dev/yolo.bin").unwrap();

		// let mut frame_iter = FrameIter::try_from(InitInfo {
		// 	source_path: PathBuf::from_str("rtsp://localhost:8554/mystream").unwrap(),
		// 	resize_info: Dim { w: WIDTH as i32, h: HEIGHT as i32 },
		// 	is_rtsp: true,
		// })
		// .unwrap();

		let mut frame_iter = CpuFrameIter::try_from(InitInfo {
			source_path: PathBuf::from_str("/home/satyam/dev/videos/cats.mp4").unwrap(),
			resize_info: Dim { w: WIDTH as i32, h: HEIGHT as i32 },
			is_rtsp: false,
		})
		.unwrap();

		let mut frame_cnt: i32 = 1;

		while let Some(frame) = frame_iter.next() {
			let input_tensor: TensorRefMut<'_, f32> = unsafe {
				TensorRefMut::from_raw(
					MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
					frame.ptr as *mut ort_sys::c_void,
					Shape::new([1, 3, HEIGHT as i64, WIDTH as i64]),
				)
				.unwrap()
			};

			let outputs = session.run(ort::inputs![input_tensor])?;
			let result = post_process(WIDTH as u32, HEIGHT as u32, outputs)?;
			println!("{:?}", result);
			println!("{frame_cnt}");
			frame_cnt += 1;

			// // this was for GPU raw frame from hardware decoder
			// // we have a CuDevicePtr available. Need to process it.
			// let rgb_size = WIDTH * HEIGHT * 3; // RGB size (3 bytes per pixel)

			// // will store rgb frame here
			// let mut rgb_ptr: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros(rgb_size as usize).unwrap();

			// let mut builder = stream.launch_builder(&f);
			// builder.arg(&frame.ptr);
			// builder.arg(&mut rgb_ptr);
			// builder.arg(&WIDTH);
			// builder.arg(&HEIGHT);
			// builder.arg(&(WIDTH * 3 * std::mem::size_of::<f32>() as u32));

			// let block_size = (16, 16, 1);
			// let grid_size = ((WIDTH + block_size.0 - 1) / block_size.0, (HEIGHT + block_size.1 - 1) / block_size.1, 1);
			// let cfg = LaunchConfig {
			// 	grid_dim: grid_size,
			// 	block_dim: block_size,
			// 	shared_mem_bytes: 0,
			// };
			// unsafe { builder.launch(cfg) }?;

			// // // will store preprocessed CHW format frame here
			// let mut final_ptr: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros(rgb_size as usize).unwrap();
			// let mut builder = stream.launch_builder(&g);
			// builder.arg(&rgb_ptr);
			// builder.arg(&mut final_ptr);
			// builder.arg(&WIDTH);
			// builder.arg(&HEIGHT);

			// let block_size = (16, 16, 1);
			// let grid_size = ((WIDTH + block_size.0 - 1) / block_size.0, (HEIGHT + block_size.1 - 1) / block_size.1, 1);
			// let cfg = LaunchConfig {
			// 	grid_dim: grid_size,
			// 	block_dim: block_size,
			// 	shared_mem_bytes: 0,
			// };
			// unsafe { builder.launch(cfg) }?;
			// let input_tensor: TensorRefMut<'_, f32> = unsafe {
			// 	TensorRefMut::from_raw(
			// 		MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
			// 		final_ptr.device_ptr(&stream).0 as *mut ort_sys::c_void,
			// 		Shape::new([1, 3, HEIGHT as i64, WIDTH as i64]),
			// 	)
			// 	.unwrap()
			// };

			// let outputs = session.run(ort::inputs![input_tensor])?;
			// // let result = post_process(frame_iter.original_width as u32, frame_iter.original_height as u32, outputs)?;
			// let result = post_process(WIDTH as u32, HEIGHT as u32, outputs)?;
			// println!("{:?}", result);
			// println!("{frame_cnt}");
			// frame_cnt += 1;
		}
	} else {
		let img_path = "/home/satyam/dev/ort/examples/cudarc/data/car2.png";
		let original_img: image::DynamicImage = image::open(Path::new(img_path)).unwrap();
		let (img_width, img_height) = (original_img.width(), original_img.height());
		println!("Original image size: {img_width} and {img_height}");
		let img = original_img.resize_exact(WIDTH, HEIGHT, FilterType::Triangle);
		let mut input = Array::zeros((1, 3, HEIGHT as usize, WIDTH as usize));
		for pixel in img.pixels() {
			let x = pixel.0 as _;
			let y = pixel.1 as _;
			let [r, g, b, _] = pixel.2.0;
			input[[0, 0, y, x]] = (r as f32) / 255.;
			input[[0, 1, y, x]] = (g as f32) / 255.;
			input[[0, 2, y, x]] = (b as f32) / 255.;
		}

		let size = 3 * WIDTH * HEIGHT * 4;
		println!("The size is : {size}");

		let mut dev_ptr: *mut c_void = std::ptr::null_mut();
		unsafe {
			cudarc::runtime::sys::cudaMalloc(&mut dev_ptr, size as usize);
		}
		unsafe {
			cudarc::runtime::result::memcpy_htod_sync(dev_ptr, &input.into_raw_vec());
		}
		let tensor: TensorRefMut<'_, f32> = unsafe {
			TensorRefMut::from_raw(
				MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
				dev_ptr as *mut ort_sys::c_void,
				Shape::new([1, 3, HEIGHT as i64, WIDTH as i64]),
			)
			.unwrap()
		};
		let outputs = session.run(ort::inputs![tensor])?;
		println!("{:?}", outputs);

		let result = post_process(img_width, img_height, outputs)?;

		println!("{:?}", result);
		let dyn_img = open(img_path).expect("Failed to open image");
		let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = dyn_img.to_rgba8();

		let color = Rgba([0, 255, 0, 255]);

		for det in result {
			let BoundingBox { x1, y1, x2, y2 } = det.0;
			let base_x = x1 as i32;
			let base_y = y1 as i32;
			let width = (x2 - x1) as u32;
			let height = (y2 - y1) as u32;

			// Increase border thickness by drawing multiple rectangles
			let thickness = 5; // You can increase this for thicker borders

			for offset in 0..thickness {
				let rect = imageproc::rect::Rect::at(base_x - offset, base_y - offset).of_size(width + (offset * 2) as u32, height + (offset * 2) as u32);
				draw_hollow_rect_mut(&mut img, rect, color);
			}
		}

		let output_path = "/home/satyam/dev/x-cuda.png";
		img.save(output_path).expect("Failed to save image");
	}

	Ok(())
}

fn post_process<'a>(img_width: u32, img_height: u32, outputs: ort::session::SessionOutputs<'a, 'a>) -> Result<Vec<(BoundingBox, &'a str, f32)>, anyhow::Error> {
	let output = outputs["output0"].try_extract_array::<f32>()?.t().into_owned();
	// println!("{:?}", output);
	let mut boxes = Vec::new();
	let output = output.slice(s![.., .., 0]);
	for row in output.axis_iter(Axis(0)) {
		let row: Vec<_> = row.iter().copied().collect();
		let (class_id, prob) = row
			    .iter()
			    // skip bounding box coordinates
			    .skip(4)
			    .enumerate()
			    .map(|(index, value)| (index, *value))
			    .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
			    .unwrap();
		if prob < 0.1 {
			continue;
		}
		let label = YOLOV8_CLASS_LABELS[class_id];
		let xc = row[0] / 640. * (img_width as f32);
		let yc = row[1] / 640. * (img_height as f32);
		let w = row[2] / 640. * (img_width as f32);
		let h = row[3] / 640. * (img_height as f32);
		boxes.push((
			BoundingBox {
				x1: xc - w / 2.,
				y1: yc - h / 2.,
				x2: xc + w / 2.,
				y2: yc + h / 2.,
			},
			label,
			prob,
		));
	}
	boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
	let mut result = Vec::new();
	while !boxes.is_empty() {
		result.push(boxes[0]);
		boxes = boxes
			.iter()
			.filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7)
			.copied()
			.collect();
	}
	Ok(result)
}

fn ffmpeg_id_to_nv_id(codec_id: Id) -> nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec {
	match codec_id {
		Id::MPEG1VIDEO => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_MPEG1,
		Id::MPEG2VIDEO => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_MPEG2,
		Id::MPEG4 => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_MPEG4,
		Id::WMV3 | Id::VC1 => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_VC1,
		Id::H264 => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_H264,
		Id::HEVC => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_HEVC,
		Id::VP8 => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_VP8,
		Id::VP9 => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_VP9,
		Id::MJPEG => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_JPEG,
		Id::AV1 => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_AV1,
		_ => nvidia_video_codec_sdk::sys::cuviddec::cudaVideoCodec::cudaVideoCodec_NumCodecs,
	}
}
