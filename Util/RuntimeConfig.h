#pragma once

#include <string>

struct RuntimeConfig {
	std::string data_root = "./input-data";
	std::string graph_root = "";
	std::string sam_root = "";
	std::string depth_root = "";
	std::string output_root = "./input-data";
	std::string method = "obj-gsp";
	double content_preserving_weight = 1.5;
	double depth_tau = 0.12;
	double depth_cross_layer_weight = 0.35;
	double depth_min_weight = 0.20;
	double depth_confidence_floor = 1.0;
	double depth_structure_weight = 1.0;
	double depth_texture_weight = 0.35;
	double depth_edge_weight = 0.50;
	double depth_texture_noise_weight = 0.35;
	double depth_planarity_weight = 0.0;
	double max_target_megapixels = 80.0;
};

extern RuntimeConfig g_runtime_config;

std::string normalizeMethod(std::string method);
std::string normalizePath(std::string path);
std::string joinPath(const std::string& left, const std::string& right);
void ensureDirectory(const std::string& path);
bool isGspMethod();
bool isDepthMethod();
bool isSamMethod();
bool isKnownMethod();
bool usesContentPreservingTerm();
std::string resultSuffix();
