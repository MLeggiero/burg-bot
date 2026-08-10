cmake_minimum_required(VERSION 3.5)
project(burgerbot_description)


find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

install(
  DIRECTORY launch meshes urdf rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

# Third-party AWS RoboMaker assets for small_house / small_warehouse worlds.
# Not vendored into the repo (~117MB of binaries) -- fetch them with
# scripts/fetch_sim_assets.sh. The empty world works without them.
foreach(asset_dir models photos)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${asset_dir}")
    install(DIRECTORY ${asset_dir} DESTINATION share/${PROJECT_NAME})
  else()
    message(STATUS
      "burgerbot_description: '${asset_dir}/' absent -- the furnished Gazebo "
      "worlds will load empty. Run scripts/fetch_sim_assets.sh to populate.")
  endif()
endforeach()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()