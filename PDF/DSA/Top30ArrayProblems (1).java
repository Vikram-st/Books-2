import java.util.*;
public class Top30ArrayProblems {

    // 1. Two Sum
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int diff = target - nums[i];
            if (map.containsKey(diff)) return new int[]{map.get(diff), i};
            map.put(nums[i], i);
        }
        return new int[]{};
    }

    // 2. 3Sum
    public static List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length-2; i++) {
            if (i>0 && nums[i]==nums[i-1]) continue;
            int l=i+1, r=nums.length-1;
            while (l<r) {
                int sum = nums[i]+nums[l]+nums[r];
                if (sum==0) {
                    res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while (l<r && nums[l]==nums[l+1]) l++;
                    while (l<r && nums[r]==nums[r-1]) r--;
                    l++; r--;
                } else if (sum<0) l++; else r--;
            }
        }
        return res;
    }

    // 3. 4Sum
    public static List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n-3; i++) {
            if (i>0 && nums[i]==nums[i-1]) continue;
            for (int j=i+1;j<n-2;j++) {
                if (j>i+1 && nums[j]==nums[j-1]) continue;
                int l=j+1, r=n-1;
                while (l<r) {
                    long sum=(long)nums[i]+nums[j]+nums[l]+nums[r];
                    if (sum==target) {
                        res.add(Arrays.asList(nums[i],nums[j],nums[l],nums[r]));
                        while(l<r && nums[l]==nums[l+1]) l++;
                        while(l<r && nums[r]==nums[r-1]) r--;
                        l++; r--;
                    } else if (sum<target) l++; else r--;
                }
            }
        }
        return res;
    }

    // 4. Longest Consecutive Sequence
    public static int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int n: nums) set.add(n);
        int best=0;
        for (int n: set) {
            if (!set.contains(n-1)) {
                int len=1;
                while (set.contains(n+len)) len++;
                best=Math.max(best,len);
            }
        }
        return best;
    }

    // 5. Majority Element
    public static int majorityElement(int[] nums) {
        int count=0, candidate=0;
        for (int n: nums) {
            if (count==0) candidate=n;
            count += (n==candidate)?1:-1;
        }
        return candidate;
    }

    // 6. Majority Element II
    public static List<Integer> majorityElementII(int[] nums) {
        int c1=0,c2=0,v1=0,v2=0;
        for (int n: nums) {
            if (n==v1) c1++;
            else if (n==v2) c2++;
            else if (c1==0) {v1=n;c1=1;}
            else if (c2==0) {v2=n;c2=1;}
            else {c1--;c2--;}
        }
        List<Integer> res = new ArrayList<>();
        c1=c2=0;
        for (int n: nums) {
            if (n==v1) c1++;
            else if (n==v2) c2++;
        }
        if (c1>nums.length/3) res.add(v1);
        if (c2>nums.length/3) res.add(v2);
        return res;
    }

    // 7. Next Permutation
    public static void nextPermutation(int[] nums) {
        int i=nums.length-2;
        while(i>=0 && nums[i]>=nums[i+1]) i--;
        if (i>=0) {
            int j=nums.length-1;
            while(nums[j]<=nums[i]) j--;
            int tmp=nums[i];nums[i]=nums[j];nums[j]=tmp;
        }
        for (int l=i+1,r=nums.length-1;l<r;l++,r--) {
            int tmp=nums[l];nums[l]=nums[r];nums[r]=tmp;
        }
    }

    // 8. Rotate Array
    public static void rotate(int[] nums, int k) {
        int n=nums.length;
        k%=n;
        reverse(nums,0,n-1);
        reverse(nums,0,k-1);
        reverse(nums,k,n-1);
    }
    private static void reverse(int[] nums,int l,int r){
        while(l<r){int tmp=nums[l];nums[l]=nums[r];nums[r]=tmp;l++;r--;}
    }

    // 9. Product of Array Except Self
    public static int[] productExceptSelf(int[] nums) {
        int n=nums.length;
        int[] res=new int[n];
        Arrays.fill(res,1);
        int prefix=1;
        for (int i=0;i<n;i++){res[i]=prefix;prefix*=nums[i];}
        int suffix=1;
        for (int i=n-1;i>=0;i--){res[i]*=suffix;suffix*=nums[i];}
        return res;
    }

    // 10. Maximum Subarray Sum
    public static int maxSubArray(int[] nums) {
        int curr=nums[0], best=nums[0];
        for (int i=1;i<nums.length;i++){curr=Math.max(nums[i],curr+nums[i]);best=Math.max(best,curr);}
        return best;
    }

    // 11. Maximum Product Subarray
    public static int maxProduct(int[] nums) {
        int max=nums[0],min=nums[0],res=nums[0];
        for(int i=1;i<nums.length;i++){
            if(nums[i]<0){int tmp=max;max=min;min=tmp;}
            max=Math.max(nums[i],max*nums[i]);
            min=Math.min(nums[i],min*nums[i]);
            res=Math.max(res,max);
        }
        return res;
    }

    // 12. Find Minimum in Rotated Sorted Array
    public static int findMin(int[] nums) {
        int l=0,r=nums.length-1;
        while(l<r){
            int m=(l+r)/2;
            if(nums[m]>nums[r]) l=m+1; else r=m;
        }
        return nums[l];
    }

    // 13. Search in Rotated Sorted Array
    public static int search(int[] nums,int target){
        int l=0,r=nums.length-1;
        while(l<=r){
            int m=(l+r)/2;
            if(nums[m]==target) return m;
            if(nums[l]<=nums[m]){
                if(nums[l]<=target && target<nums[m]) r=m-1; else l=m+1;
            }else{
                if(nums[m]<target && target<=nums[r]) l=m+1; else r=m-1;
            }
        }
        return -1;
    }

    // 14. Find the Duplicate Number (Floyd’s cycle)
    public static int findDuplicate(int[] nums){
        int slow=nums[0],fast=nums[0];
        do{slow=nums[slow];fast=nums[nums[fast]];}while(slow!=fast);
        slow=nums[0];
        while(slow!=fast){slow=nums[slow];fast=nums[fast];}
        return slow;
    }

    // 15. Set Matrix Zeroes
    public static void setZeroes(int[][] matrix){
        int m=matrix.length,n=matrix[0].length;
        boolean fr=false,fc=false;
        for(int i=0;i<m;i++) if(matrix[i][0]==0) fc=true;
        for(int j=0;j<n;j++) if(matrix[0][j]==0) fr=true;
        for(int i=1;i<m;i++) for(int j=1;j<n;j++) if(matrix[i][j]==0){matrix[i][0]=0;matrix[0][j]=0;}
        for(int i=1;i<m;i++) for(int j=1;j<n;j++) if(matrix[i][0]==0||matrix[0][j]==0) matrix[i][j]=0;
        if(fc) for(int i=0;i<m;i++) matrix[i][0]=0;
        if(fr) for(int j=0;j<n;j++) matrix[0][j]=0;
    }

    // 16. Spiral Matrix
    public static List<Integer> spiralOrder(int[][] matrix){
        List<Integer> res=new ArrayList<>();
        if(matrix.length==0) return res;
        int top=0,bottom=matrix.length-1,left=0,right=matrix[0].length-1;
        while(top<=bottom && left<=right){
            for(int j=left;j<=right;j++) res.add(matrix[top][j]); top++;
            for(int i=top;i<=bottom;i++) res.add(matrix[i][right]); right--;
            if(top<=bottom){for(int j=right;j>=left;j--) res.add(matrix[bottom][j]); bottom--;}
            if(left<=right){for(int i=bottom;i>=top;i--) res.add(matrix[i][left]); left++;}
        }
        return res;
    }

    // 17. Merge Intervals
    public static int[][] merge(int[][] intervals){
        Arrays.sort(intervals,(a,b)->a[0]-b[0]);
        List<int[]> res=new ArrayList<>();
        int[] curr=intervals[0];
        for(int i=1;i<intervals.length;i++){
            if(intervals[i][0]<=curr[1]) curr[1]=Math.max(curr[1],intervals[i][1]);
            else{res.add(curr);curr=intervals[i];}
        }
        res.add(curr);
        return res.toArray(new int[res.size()][]);
    }

    // 18. Insert Interval
    public static int[][] insert(int[][] intervals,int[] newInterval){
        List<int[]> res=new ArrayList<>();
        int i=0,n=intervals.length;
        while(i<n && intervals[i][1]<newInterval[0]) res.add(intervals[i++]);
        while(i<n && intervals[i][0]<=newInterval[1]){
            newInterval[0]=Math.min(newInterval[0],intervals[i][0]);
            newInterval[1]=Math.max(newInterval[1],intervals[i][1]);
            i++;
        }
        res.add(newInterval);
        while(i<n) res.add(intervals[i++]);
        return res.toArray(new int[res.size()][]);
    }

    // 19. Interval List Intersections
    public static int[][] intervalIntersection(int[][] A,int[][] B){
        List<int[]> res=new ArrayList<>();
        int i=0,j=0;
        while(i<A.length && j<B.length){
            int lo=Math.max(A[i][0],B[j][0]);
            int hi=Math.min(A[i][1],B[j][1]);
            if(lo<=hi) res.add(new int[]{lo,hi});
            if(A[i][1]<B[j][1]) i++; else j++;
        }
        return res.toArray(new int[res.size()][]);
    }

    // 20. Subarray Sum Equals K
    public static int subarraySum(int[] nums,int k){
        Map<Integer,Integer> map=new HashMap<>();
        map.put(0,1);
        int sum=0,count=0;
        for(int n:nums){
            sum+=n;
            count+=map.getOrDefault(sum-k,0);
            map.put(sum,map.getOrDefault(sum,0)+1);
        }
        return count;
    }

    // 21. Minimum Size Subarray Sum
    public static int minSubArrayLen(int target,int[] nums){
        int l=0,sum=0,res=Integer.MAX_VALUE;
        for(int r=0;r<nums.length;r++){
            sum+=nums[r];
            while(sum>=target){
                res=Math.min(res,r-l+1);
                sum-=nums[l++];
            }
        }
        return res==Integer.MAX_VALUE?0:res;
    }

    // 22. Longest Substring Without Repeating Characters
    public static int lengthOfLongestSubstring(String s){
        Map<Character,Integer> map=new HashMap<>();
        int l=0,res=0;
        for(int r=0;r<s.length();r++){
            if(map.containsKey(s.charAt(r))) l=Math.max(l,map.get(s.charAt(r))+1);
            map.put(s.charAt(r),r);
            res=Math.max(res,r-l+1);
        }
        return res;
    }

    // 23. Container With Most Water
    public static int maxArea(int[] height){
        int l=0,r=height.length-1,res=0;
        while(l<r){
            res=Math.max(res,Math.min(height[l],height[r])*(r-l));
            if(height[l]<height[r]) l++; else r--;
        }
        return res;
    }

    // 24. Trapping Rain Water
    public static int trap(int[] height){
        int l=0,r=height.length-1,lmax=0,rmax=0,res=0;
        while(l<r){
            if(height[l]<height[r]){
                if(height[l]>=lmax) lmax=height[l]; else res+=lmax-height[l];
                l++;
            }else{
                if(height[r]>=rmax) rmax=height[r]; else res+=rmax-height[r];
                r--;
            }
        }
        return res;
    }

    // 25. Jump Game
    public static boolean canJump(int[] nums){
        int reach=0;
        for(int i=0;i<nums.length;i++){
            if(i>reach) return false;
            reach=Math.max(reach,i+nums[i]);
        }
        return true;
    }

    // 26. Jump Game II
    public static int jump(int[] nums){
        int jumps=0,currEnd=0,currFarthest=0;
        for(int i=0;i<nums.length-1;i++){
            currFarthest=Math.max(currFarthest,i+nums[i]);
            if(i==currEnd){jumps++;currEnd=currFarthest;}
        }
        return jumps;
    }

    // 27. Sort Colors
    public static void sortColors(int[] nums){
        int l=0,r=nums.length-1,i=0;
        while(i<=r){
            if(nums[i]==0){int tmp=nums[l];nums[l]=nums[i];nums[i]=tmp;l++;i++;}
            else if(nums[i]==2){int tmp=nums[r];nums[r]=nums[i];nums[i]=tmp;r--;}
            else i++;
        }
    }

    // 28. Find First and Last Position of Element in Sorted Array
    public static int[] searchRange(int[] nums,int target){
        return new int[]{first(nums,target),last(nums,target)};
    }
    private static int first(int[] nums,int target){
        int l=0,r=nums.length-1,res=-1;
        while(l<=r){int m=(l+r)/2;if(nums[m]>=target){if(nums[m]==target) res=m;r=m-1;}else l=m+1;}return res;
    }
    private static int last(int[] nums,int target){
        int l=0,r=nums.length-1,res=-1;
        while(l<=r){int m=(l+r)/2;if(nums[m]<=target){if(nums[m]==target) res=m;l=m+1;}else r=m-1;}return res;
    }

    // 29. Find Peak Element
    public static int findPeakElement(int[] nums){
        int l=0,r=nums.length-1;
        while(l<r){int m=(l+r)/2;if(nums[m]>nums[m+1]) r=m; else l=m+1;}return l;
    }

    // 30. Minimum Swaps to Group All 1’s Together
    public static int minSwaps(int[] data){
        int ones=0;for(int n:data) if(n==1) ones++;
        int curr=0;for(int i=0;i<ones;i++) if(data[i]==1) curr++;
        int max=curr;
        for(int i=ones;i<data.length;i++){
            if(data[i]==1) curr++;
            if(data[i-ones]==1) curr--;
            max=Math.max(max,curr);
        }
        return ones-max;
    }

    public static void main(String[] args){
        System.out.println("Two Sum: "+Arrays.toString(twoSum(new int[]{2,7,11,15},9)));
        System.out.println("3Sum: "+threeSum(new int[]{-1,0,1,2,-1,-4}));
        System.out.println("Find Min in Rotated Array: "+findMin(new int[]{4,5,6,7,0,1,2}));
        System.out.println("Max SubArray: "+maxSubArray(new int[]{-2,1,-3,4,-1,2,1,-5,4}));
        System.out.println("Trap Rain Water: "+trap(new int[]{0,1,0,2,1,0,1,3,2,1,2,1}));
    }
}
