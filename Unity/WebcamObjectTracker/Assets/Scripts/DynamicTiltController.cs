using UnityEngine;
using System.Collections.Generic;

public class DynamicTiltController : MonoBehaviour
{
    [Header("기울기 설정")]
    public float tiltSpeed = 10f;          // 기울기 속도
    public float maxTiltAngle = 15f;       // 최대 기울기 각도
    public float objectInfluence = 1.0f;   // 객체의 무게 영향력
    public Vector2 boardSize = new Vector2(0.15f, 0.15f); // 보드 크기 (미터)
    
    [Header("참조")]
    public Transform platformTransform;    // 플랫폼(원판) 트랜스폼
    
    private Rigidbody rb;
    private UDPReceiver udpReceiver;
    private Vector2 targetTilt = Vector2.zero;

    void Start()
    {
        // Rigidbody 컴포넌트 가져오기
        rb = GetComponent<Rigidbody>();
        
        // 만약 platformTransform이 할당되지 않았다면 현재 오브젝트 사용
        if (platformTransform == null)
            platformTransform = transform;
        
        // UDPReceiver 찾기 및 이벤트 구독
        udpReceiver = FindObjectOfType<UDPReceiver>();
        if (udpReceiver != null)
        {
            udpReceiver.OnObjectsDetected += OnObjectsDetected;
            Debug.Log("UDPReceiver에 연결됨");
        }
        else
        {
            Debug.LogError("UDPReceiver를 찾을 수 없습니다!");
        }
    }
    
    // 감지된 객체 정보에 따라 타겟 기울기 계산
    void OnObjectsDetected(List<DetectedObject> objects)
    {
        if (objects.Count == 0)
            return;
            
        float xTiltSum = 0f;
        float zTiltSum = 0f;
        float totalWeight = 0f;
        
        foreach (var obj in objects)
        {
            // 정규화된 위치 (0~1)를 중앙 기준 위치(-0.5~0.5)로 변환
            float normalizedX = obj.norm_x - 0.5f;
            float normalizedY = obj.norm_y - 0.5f;
            
            // 신뢰도를 무게로 사용 (옵션)
            float weight = obj.conf;
            
            // X축 기울기 (앞/뒤)는 Y값에 영향 받음
            xTiltSum += normalizedY * weight;
            
            // Z축 기울기 (좌/우)는 X값에 영향 받음 (부호 반전)
            zTiltSum += -normalizedX * weight;
            
            totalWeight += weight;
        }
        
        // 무게 중심에 따른 평균 기울기 계산
        if (totalWeight > 0)
        {
            targetTilt.x = (xTiltSum / totalWeight) * maxTiltAngle * objectInfluence;
            targetTilt.y = (zTiltSum / totalWeight) * maxTiltAngle * objectInfluence;
        }
    }
    
    void Update()
    {
        // 키보드 입력으로 수동 제어 (테스트용)
        float horizontalInput = Input.GetAxis("Horizontal");
        float verticalInput = Input.GetAxis("Vertical");
        
        if (Mathf.Abs(horizontalInput) > 0.1f || Mathf.Abs(verticalInput) > 0.1f)
        {
            // 수동 입력으로 타겟 기울기 덮어쓰기
            targetTilt = new Vector2(
                verticalInput * maxTiltAngle,
                -horizontalInput * maxTiltAngle
            );
        }
        
        // 목표 회전 각도 계산
        Vector3 targetRotation = new Vector3(
            targetTilt.x,  // X축 회전 (앞/뒤 기울기)
            0,             // Y축 회전 없음
            targetTilt.y   // Z축 회전 (좌/우 기울기)
        );

        // 부드럽게 회전 적용
        platformTransform.rotation = Quaternion.Slerp(
            platformTransform.rotation,
            Quaternion.Euler(targetRotation),
            Time.deltaTime * tiltSpeed
        );
    }
    
    // GUI에 기울기 정보 표시 (디버그용)
    void OnGUI()
    {
        GUI.backgroundColor = Color.black;
        GUI.contentColor = Color.white;
        
        GUILayout.BeginArea(new Rect(10, 200, 300, 100));
        GUILayout.Label($"기울기: X={targetTilt.x:F1}°, Z={targetTilt.y:F1}°");
        GUILayout.EndArea();
    }
}