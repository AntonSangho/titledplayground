using UnityEngine;
using System;

public static class JsonHelper
{
    // JSON 배열을 객체 배열로 변환
    public static T[] FromJson<T>(string json)
    {
        // JSON 배열을 래핑하여 JsonUtility가 처리할 수 있게 함
        string newJson = "{ \"objects\": " + json + "}";
        Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(newJson);
        return wrapper.objects;
    }

    // 객체 배열을 JSON으로 변환
    public static string ToJson<T>(T[] array, bool prettyPrint = false)
    {
        Wrapper<T> wrapper = new Wrapper<T>();
        wrapper.objects = array;
        return JsonUtility.ToJson(wrapper, prettyPrint);
    }

    // 직렬화/역직렬화를 위한 래퍼 클래스
    [Serializable]
    private class Wrapper<T>
    {
        public T[] objects;
    }
}