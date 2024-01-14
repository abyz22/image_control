BMAB와 다르게 아웃패딩된 부분'만' 마스킹하지않고 이미지 전체를 I2I돌림,
인물 작아질때 새로 그리는걸 방지하기 위해 컨트롤넷 사용(OpenPose)
+ 확대했을때도 추가 (0.3x ~ 2.5x)
+ 이미지 각각 다른 배율 설정



### Controlnet name: **open pose로 설정**
###### pose strength : control net strength
###### pad_mode: outpainting mode (확대시 사용X)
###### mode_type : 위치설정
###### ratio_min,ratio_max : 축소/확대비율

### sample image:
![ComfyUI_temp_jxqnh_00015_](https://github.com/ThisisLandu/landu_outpainting/assets/36629328/d961dd12-d4d4-42e3-8299-a7f6c1165176)
