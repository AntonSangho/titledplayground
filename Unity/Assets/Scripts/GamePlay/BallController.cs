using UnityEngine;

public class BallController : MonoBehaviour
{
    private Rigidbody rb;
    private Vector3 startPosition;
    
    void Start()
    {
        // Rigidbody 컴포넌트 참조 가져오기
        rb = GetComponent<Rigidbody>();
        
        // 시작 위치 저장
        startPosition = transform.position;
    }
    
    void FixedUpdate()
    {
        // 만약 공이 떨어졌다면, 다시 시작 위치로 되돌리는 코드
        if (transform.position.y < -10f)
        {
            ResetBall();
        }
    }
    
    void ResetBall()
    {
        // 공을 원래 위치로 재설정
        transform.position = startPosition;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }
}